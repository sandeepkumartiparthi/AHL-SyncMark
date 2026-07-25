import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import os

from dataset import get_dataloaders
from model import ResNetHost
from watermark_core import (
    compute_fim_channels,
    train_ahl_decoder,
    evaluate_ahl_match,
    compute_smoothgrad_ig,
    pearson_correlation,
    train_eaaw
)
from fama_d import (
    generate_csk_projection,
    CSKActivationWrapper,
    CLADAMapping,
    train_clada,
    evaluate_clada_match,
    distill_student_model
)
from zk_registry import ZkSNARKRegistry

def apply_structured_pruning(model, amount=0.5):

    pruned_model = copy.deepcopy(model)
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            l1_norms = weight.abs().sum(dim=(1, 2, 3))
            num_to_prune = int(len(l1_norms) * amount)
            if num_to_prune > 0:
                threshold = torch.kthvalue(l1_norms, num_to_prune).values
                mask = l1_norms > threshold
                for i in range(len(mask)):
                    if not mask[i]:
                        weight[i] = 0.0
    return pruned_model

def apply_simulated_quantization(model):

    quant_model = copy.deepcopy(model)
    for name, param in quant_model.named_parameters():
        if param.requires_grad:
            w = param.data
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / 254.0 + 1e-8
            zero_point = 127.0 - w_max / scale

            w_quant = torch.round(w / scale + zero_point)
            w_quant = torch.clamp(w_quant, 0, 254)

            w_dequant = (w_quant - zero_point) * scale
            param.data.copy_(w_dequant)
    return quant_model

def train_base_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training host model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    return model

def evaluate_cacc(model, loader, device):

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Generating synthetic datasets...")
    train_loader, val_loader = get_dataloaders(batch_size=128, train_samples=3000, val_samples=800, seed=42)

    model = ResNetHost(num_classes=10).to(device)
    model = train_base_model(model, train_loader, val_loader, epochs=8, lr=1e-3, device=device)

    baseline_cacc = evaluate_cacc(model, val_loader, device)
    print(f"Baseline Clean Classification Accuracy (CACC): {baseline_cacc*100:.2f}%")

    print("\n--- Step 3: FIM-guided Stable Channel Selection ---")
    fim = compute_fim_channels(model, train_loader, device=device)

    sorted_channels = torch.argsort(fim, descending=True).tolist()

    k = 25
    stable_channels_base = sorted_channels[:k]
    print(f"Top 5 stable channels (FIM): {stable_channels_base[:5]}")

    stable_channels_fama = sorted_channels[:3*k]

    signature = np.random.choice([-1.0, 1.0], size=32).tolist()
    print(f"Private Ownership Signature (First 8 bits): {signature[:8]}")

    print("\n--- Step 4: Embedding Baseline AHL Decoder (Gradient-Isolated) ---")
    decoder_base = train_ahl_decoder(
        model, stable_channels_base, signature, train_loader, epochs=5, lr=1e-2, device=device
    )

    base_match = evaluate_ahl_match(model, decoder_base, stable_channels_base, signature, val_loader, device)
    print(f"Baseline AHL Signature Match Rate (No Attack): {base_match*100:.2f}%")
    print(f"CACC after Baseline AHL embedding: {evaluate_cacc(model, val_loader, device)*100:.2f}% (Gradient isolated -> 0.00% drop)")

    print("\n--- Step 5: Explanation-as-a-Watermark (EaaW) Channel ---")
    trigger_img, trigger_label = next(iter(val_loader))
    trigger_img = trigger_img[0:1]
    target_class = trigger_label[0].item()

    target_mask = torch.zeros((1, 3, 64, 64), device=device)
    target_mask[:, :, 20:44, 20:44] = 1.0

    print("Embedding EaaW watermark...")
    watermarked_model = copy.deepcopy(model)
    watermarked_model = train_eaaw(
        watermarked_model, trigger_img, target_mask, target_class,
        train_loader, val_loader, epochs=3, lr=1e-4, device=device
    )

    attr_original = compute_smoothgrad_ig(
        watermarked_model, trigger_img, target_class, steps=5, N=5, sigma=0.1, device=device
    )
    eaaw_similarity_no_attack = pearson_correlation(attr_original, target_mask).item()
    print(f"EaaW Attribution Similarity (No Attack): {eaaw_similarity_no_attack*100:.2f}%")

    watermarked_cacc = evaluate_cacc(watermarked_model, val_loader, device)
    print(f"CACC after EaaW Watermark: {watermarked_cacc*100:.2f}% (Drop: {(baseline_cacc - watermarked_cacc)*100:.2f}%)")

    print("\n--- Step 6: zk-SNARK Blockchain Registry Simulation ---")
    registry = ZkSNARKRegistry()
    h_m, h_s = registry.commit_model(watermarked_model, signature)
    print(f"Committed Model Hash (H_M): {h_m[:32]}...")
    print(f"Committed Signature Hash (H_S): {h_s[:32]}...")

    proof = registry.generate_proof(
        watermarked_model, decoder_base, stable_channels_base, signature, h_m, h_s, trigger_img
    )
    verification_success = registry.verify_proof(h_m, h_s, proof)
    print(f"zk-SNARK On-Chain Verification: {'SUCCESS' if verification_success else 'FAILED'}")

    print("\n--- Step 7: FAMA-D Implementation (CLADA + CSK) ---")
    private_key = 1337
    R_CSK = generate_csk_projection(k, 3*k, private_key=private_key)
    csk_wrapper = CSKActivationWrapper(stable_channels_fama, R_CSK, device)

    def evaluate_csk_match(model, decoder, wrapper, signature, loader, device):
        model.eval()
        decoder.eval()
        correct = 0
        total = 0
        sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for x, _ in loader:
                x = x.to(device)
                _, act = model(x, return_activations=True)
                h_proj = wrapper.project(act)
                pred_sig = decoder(h_proj)
                pred_sign = torch.sign(pred_sig)
                pred_sign[pred_sign == 0] = -1
                correct += (pred_sign == sig_tensor.unsqueeze(0)).sum().item()
                total += x.size(0) * len(signature)
        return correct / total

    decoder_csk = nn.Sequential(
        nn.Linear(k, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, len(signature))
    ).to(device)

    opt_csk = torch.optim.Adam(decoder_csk.parameters(), lr=1e-2)
    sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device).unsqueeze(0)

    print("Training FAMA-D AHL Decoder (under CSK projection)...")
    watermarked_model.eval()
    for epoch in range(5):
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                _, act = watermarked_model(x, return_activations=True)
                h_proj = csk_wrapper.project(act)
            pred = decoder_csk(h_proj)
            loss = F.mse_loss(pred, sig_tensor.expand_as(pred))
            opt_csk.zero_grad()
            loss.backward()
            opt_csk.step()

    csk_match = evaluate_csk_match(watermarked_model, decoder_csk, csk_wrapper, signature, val_loader, device)
    print(f"FAMA-D CSK Signature Match Rate (No Attack): {csk_match*100:.2f}%")

    clada_mapping = CLADAMapping(num_classes=10, d=32).to(device)
    print("Training CLADA logic (Soft-label Anchor)...")
    watermarked_model, clada_mapping = train_clada(
        watermarked_model, clada_mapping, signature, train_loader, trigger_img, epochs=4, lr=1e-3, device=device
    )
    clada_match_no_attack = evaluate_clada_match(watermarked_model, clada_mapping, signature, trigger_img, device)
    print(f"CLADA Soft-logit Signature Match Rate (No Attack): {clada_match_no_attack*100:.2f}%")

    print("\n--- Step 8: Simulated Attacks and Robustness Battery ---")
    attacks = {
        "No Attack": watermarked_model,
        "50% Structured Pruning": apply_structured_pruning(watermarked_model, amount=0.5),
        "INT8 Quantization": apply_simulated_quantization(watermarked_model),
        "Fine-Tuning (5 Epochs)": train_base_model(copy.deepcopy(watermarked_model), train_loader, val_loader, epochs=5, lr=1e-4, device=device)
    }

    print("\nSimulating Knowledge Distillation (Teacher -> Student)...")
    student = ResNetHost(num_classes=10).to(device)
    student = distill_student_model(watermarked_model, student, train_loader, trigger_img, epochs=5, lr=1e-3, temp=4.0, device=device)

    results = []

    for attack_name, attacked_model in attacks.items():
        cacc = evaluate_cacc(attacked_model, val_loader, device)

        ahl_match = evaluate_ahl_match(attacked_model, decoder_base, stable_channels_base, signature, val_loader, device)

        fama_match = evaluate_csk_match(attacked_model, decoder_csk, csk_wrapper, signature, val_loader, device)

        attr = compute_smoothgrad_ig(attacked_model, trigger_img, target_class, steps=5, N=5, sigma=0.1, device=device)
        eaaw_sim = pearson_correlation(attr, target_mask).item()

        proof = registry.generate_proof(attacked_model, decoder_base, stable_channels_base, signature, h_m, h_s, trigger_img)
        zk_status = registry.verify_proof(h_m, h_s, proof)

        clada_m = evaluate_clada_match(attacked_model, clada_mapping, signature, trigger_img, device)

        results.append({
            "Attack": attack_name,
            "CACC": cacc,
            "Baseline AHL Match": ahl_match,
            "FAMA-D CSK Match": fama_match,
            "EaaW ASR": eaaw_sim,
            "zk-SNARK": zk_status,
            "CLADA Match": clada_m
        })

    cacc_student = evaluate_cacc(student, val_loader, device)
    ahl_match_student = evaluate_ahl_match(student, decoder_base, stable_channels_base, signature, val_loader, device)
    clada_m_student = evaluate_clada_match(student, clada_mapping, signature, trigger_img, device)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"{'Attack Scenario':<25} | {'CACC':<6} | {'AHL':<6} | {'FAMA-D':<7} | {'EaaW ASR':<8} | {'zk-SNARK':<8}")
    print("-" * 75)
    for r in results:
        zk_str = "VALID" if r["zk-SNARK"] else "INVALID"
        print(f"{r['Attack']:<25} | {r['CACC']*100:5.2f}% | {r['Baseline AHL Match']*100:5.2f}% | {r['FAMA-D CSK Match']*100:6.2f}% | {r['EaaW ASR']*100:7.2f}% | {zk_str:<8}")

    print("\n================ KNOWLEDGE DISTILLATION (STUDENT MODEL) ================")
    print(f"Student Model CACC: {cacc_student*100:.2f}%")
    print(f"Baseline AHL Match Rate on Student: {ahl_match_student*100:.2f}% (Expected: poor, near 50%)")
    print(f"FAMA-D CLADA Match Rate on Student: {clada_m_student*100:.2f}% (Expected: strong transfer via soft-label distillation)")

if __name__ == "__main__":
    main()
