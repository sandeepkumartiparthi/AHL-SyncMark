import os
import time
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
from main import apply_structured_pruning, apply_simulated_quantization, evaluate_cacc

app = FastAPI(title="AHL-SyncMark & FAMA-D Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_state = {
    "status": "idle",
    "progress": 0,
    "logs": [],
    "metrics": None,
    "attribution_maps": {}
}

cache = {}

def train_base_model_local(model, train_loader, epochs=2, lr=1e-4, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def log_info(msg, progress_val=None):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    task_state["logs"].append(formatted)
    if progress_val is not None:
        task_state["progress"] = progress_val

def run_pipeline():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_info(f"Using device: {device}", 2)

        log_info("Generating synthetic datasets (shape, color, texture)...", 5)
        train_loader, val_loader = get_dataloaders(batch_size=64, train_samples=1000, val_samples=256, seed=42)
        log_info("Datasets ready.", 8)

        log_info("Initializing host ResNet classifier...", 10)
        model = ResNetHost(num_classes=10).to(device)

        epochs = 3
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        log_info(f"Starting host training for {epochs} epochs...", 12)
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
            with torch.inference_mode():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_acc = val_correct / 256

            log_info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%", 12 + int((epoch+1)/epochs * 25))

        baseline_cacc = evaluate_cacc(model, val_loader, device)
        log_info(f"Host baseline training converged. CACC: {baseline_cacc*100:.2f}%", 38)

        log_info("Computing diagonal Fisher Information Matrix (FIM)...", 40)
        fim = compute_fim_channels(model, train_loader, device=device)
        sorted_channels = torch.argsort(fim, descending=True).tolist()

        k = 25
        stable_channels_base = sorted_channels[:k]
        stable_channels_fama = sorted_channels[:3*k]
        log_info(f"Subspace channels selected. Baseline top 5: {stable_channels_base[:5]}", 45)

        signature = np.random.choice([-1.0, 1.0], size=32).tolist()
        log_info(f"Generated 32-bit private signature: {signature[:8]}...", 48)

        log_info("Training baseline AHL decoder (gradient-isolated)...", 50)
        decoder_base = train_ahl_decoder(
            model, stable_channels_base, signature, train_loader, epochs=4, lr=1e-2, device=device
        )
        base_match = evaluate_ahl_match(model, decoder_base, stable_channels_base, signature, val_loader, device)
        log_info(f"Baseline AHL Signature Match Rate: {base_match*100:.2f}% (CACC: {evaluate_cacc(model, val_loader, device)*100:.2f}%)", 55)

        log_info("Starting Explanation-as-a-Watermark (EaaW) optimization...", 60)
        trigger_img, trigger_label = next(iter(val_loader))
        trigger_img = trigger_img[0:1]
        target_class = trigger_label[0].item()

        target_mask = torch.zeros((1, 3, 64, 64), device=device)
        target_mask[:, :, 20:44, 20:44] = 1.0

        watermarked_model = copy.deepcopy(model)
        watermarked_model = train_eaaw(
            watermarked_model, trigger_img, target_mask, target_class,
            train_loader, val_loader, epochs=3, lr=1e-4, device=device
        )

        attr_orig = compute_smoothgrad_ig(
            watermarked_model, trigger_img, target_class, steps=5, N=5, sigma=0.1, device=device
        )
        eaaw_sim_no_attack = pearson_correlation(attr_orig, target_mask).item()
        log_info(f"EaaW Attribution Similarity: {eaaw_sim_no_attack*100:.2f}%", 70)

        grayscale_map = attr_orig.squeeze(0).mean(dim=0).detach().cpu().numpy().tolist()
        task_state["attribution_maps"]["watermarked"] = grayscale_map
        task_state["attribution_maps"]["target"] = target_mask.squeeze(0).mean(dim=0).detach().cpu().numpy().tolist()

        log_info("Registering model parameters and signature hashes on-chain...", 72)
        registry = ZkSNARKRegistry()
        h_m, h_s = registry.commit_model(watermarked_model, signature)
        log_info(f"Commitments published: H_M={h_m[:16]}..., H_S={h_s[:16]}...", 75)

        log_info("Applying FAMA-D CSK (Chaotic Space Keying) projection...", 78)
        R_CSK = generate_csk_projection(k, 3*k, private_key=1337)
        csk_wrapper = CSKActivationWrapper(stable_channels_fama, R_CSK, device)

        decoder_csk = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(signature))
        ).to(device)
        opt_csk = torch.optim.Adam(decoder_csk.parameters(), lr=1e-2)
        sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device).unsqueeze(0)

        for epoch in range(4):
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
        log_info("FAMA-D CSK Decoder trained successfully.", 82)

        log_info("Training FAMA-D CLADA (soft-label logit anchor)...", 85)
        clada_mapping = CLADAMapping(num_classes=10, d=32).to(device)
        watermarked_model, clada_mapping = train_clada(
            watermarked_model, clada_mapping, signature, train_loader, trigger_img, epochs=3, lr=1e-3, device=device
        )
        log_info("CLADA mapping trained successfully.", 90)

        log_info("Simulating structured attacks...", 92)

        pruned_model = apply_structured_pruning(watermarked_model, amount=0.5)
        quant_model = apply_simulated_quantization(watermarked_model)
        ft_model = train_base_model_local(copy.deepcopy(watermarked_model), train_loader, epochs=2, lr=1e-4, device=device)
        student_model = ResNetHost(num_classes=10).to(device)
        student_model = distill_student_model(watermarked_model, student_model, train_loader, trigger_img, epochs=3, lr=1e-3, temp=4.0, device=device)

        scenarios = {
            "No Attack": watermarked_model,
            "50% Structured Pruning": pruned_model,
            "INT8 Quantization": quant_model,
            "Fine-Tuning (2 Epochs)": ft_model
        }

        metrics = []
        for name, m_model in scenarios.items():
            cacc = evaluate_cacc(m_model, val_loader, device)
            ahl_match = evaluate_ahl_match(m_model, decoder_base, stable_channels_base, signature, val_loader, device)

            def eval_csk(model, decoder, wrapper, signature, loader, device):
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

            fama_match = eval_csk(m_model, decoder_csk, csk_wrapper, signature, val_loader, device)

            attr = compute_smoothgrad_ig(m_model, trigger_img, target_class, steps=5, N=5, sigma=0.1, device=device)
            eaaw_sim = pearson_correlation(attr, target_mask).item()
            task_state["attribution_maps"][name] = attr.squeeze(0).mean(dim=0).detach().cpu().numpy().tolist()

            proof = registry.generate_proof(m_model, decoder_base, stable_channels_base, signature, h_m, h_s, trigger_img)
            zk_status = registry.verify_proof(h_m, h_s, proof)

            metrics.append({
                "scenario": name,
                "cacc": float(cacc),
                "ahl": float(ahl_match),
                "fama_d": float(fama_match),
                "eaaw": float(eaaw_sim),
                "zk_snark": bool(zk_status)
            })

        student_cacc = evaluate_cacc(student_model, val_loader, device)
        student_ahl = evaluate_ahl_match(student_model, decoder_base, stable_channels_base, signature, val_loader, device)
        student_clada = evaluate_clada_match(student_model, clada_mapping, signature, trigger_img, device)

        student_metrics = {
            "cacc": float(student_cacc),
            "ahl": float(student_ahl),
            "clada": float(student_clada)
        }

        task_state["metrics"] = {
            "scenarios": metrics,
            "student": student_metrics
        }

        cache["watermarked_model"] = watermarked_model
        cache["decoder_base"] = decoder_base
        cache["stable_channels_base"] = stable_channels_base
        cache["signature"] = signature
        cache["registry"] = registry
        cache["h_m"] = h_m
        cache["h_s"] = h_s
        cache["trigger_img"] = trigger_img
        cache["target_mask"] = target_mask
        cache["target_class"] = target_class

        log_info("Pipeline execution complete! Dashboard ready.", 100)
        task_state["status"] = "done"

    except Exception as e:
        import traceback
        err_msg = f"Error: {e}\n{traceback.format_exc()}"
        log_info(err_msg)
        task_state["status"] = "error"

@app.post("/api/start-pipeline")
def start_pipeline(background_tasks: BackgroundTasks):
    if task_state["status"] == "running":
        return {"msg": "Pipeline already running."}

    task_state["status"] = "running"
    task_state["progress"] = 0
    task_state["logs"] = []
    task_state["metrics"] = None
    task_state["attribution_maps"] = {}

    background_tasks.add_task(run_pipeline)
    return {"msg": "Pipeline started."}

@app.get("/api/status")
def get_status():
    return {
        "status": task_state["status"],
        "progress": task_state["progress"],
        "logs": task_state["logs"][-30:]
    }

@app.get("/api/metrics")
def get_metrics():
    return task_state["metrics"]

@app.get("/api/attribution-map")
def get_attribution_map(name: str = "watermarked"):
    if name in task_state["attribution_maps"]:
        return {"map": task_state["attribution_maps"][name]}
    return {"map": None}

class AttackRequest(BaseModel):
    attack_type: str
    param: float

@app.post("/api/interactive-attack")
def interactive_attack(req: AttackRequest):
    if "watermarked_model" not in cache:
        return {"error": "Model not watermarked yet. Please run the pipeline first."}

    model = cache["watermarked_model"]
    decoder_base = cache["decoder_base"]
    stable_channels_base = cache["stable_channels_base"]
    signature = cache["signature"]
    registry = cache["registry"]
    h_m = cache["h_m"]
    h_s = cache["h_s"]
    trigger_img = cache["trigger_img"]
    target_mask = cache["target_mask"]
    target_class = cache["target_class"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader = get_dataloaders(batch_size=64, train_samples=1000, val_samples=256, seed=42)

    attacked = copy.deepcopy(model)

    if req.attack_type == "prune":
        attacked = apply_structured_pruning(attacked, amount=req.param)
    elif req.attack_type == "quantize":
        levels = 2 ** int(req.param)
        for name, param in attacked.named_parameters():
            if param.requires_grad:
                w = param.data
                w_min, w_max = w.min(), w.max()
                scale = (w_max - w_min) / (levels - 2) + 1e-8
                zero_point = (levels / 2) - 1.0 - w_max / scale
                w_quant = torch.round(w / scale + zero_point)
                w_quant = torch.clamp(w_quant, 0, levels - 2)
                w_dequant = (w_quant - zero_point) * scale
                param.data.copy_(w_dequant)

    cacc = evaluate_cacc(attacked, val_loader, device)

    with torch.inference_mode():
        _, act = attacked(trigger_img, return_activations=True)
        h_star = act[:, stable_channels_base].mean(dim=(2, 3))
        pred_sig = decoder_base(h_star)
        pred_sign = torch.sign(pred_sig)
        pred_sign[pred_sign == 0] = -1
        ahl_match = (pred_sign == torch.tensor(signature, device=device).unsqueeze(0)).sum().item() / len(signature)

    attr = compute_smoothgrad_ig(attacked, trigger_img, target_class, steps=5, N=5, sigma=0.1, device=device)
    eaaw_sim = pearson_correlation(attr, target_mask).item()

    proof = registry.generate_proof(attacked, decoder_base, stable_channels_base, signature, h_m, h_s, trigger_img)
    zk_status = registry.verify_proof(h_m, h_s, proof)

    return {
        "cacc": float(cacc),
        "ahl": float(ahl_match),
        "eaaw": float(eaaw_sim),
        "zk_snark": bool(zk_status),
        "map": attr.squeeze(0).mean(dim=0).detach().cpu().numpy().tolist()
    }

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
