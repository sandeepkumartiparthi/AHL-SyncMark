import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_fim_channels(model, dataloader, device):

    model.eval()
    fim_accum = torch.zeros(128, device=device)
    total_samples = 0

    criterion = nn.CrossEntropyLoss(reduction='sum')

    for idx, (x, y) in enumerate(dataloader):
        if idx >= 4:
            break
        x, y = x.to(device), y.to(device)

        logits, act = model(x, return_activations=True)
        loss = criterion(logits, y)

        grads = torch.autograd.grad(outputs=loss, inputs=act)[0]

        fim_accum += (grads ** 2).sum(dim=(0, 2, 3))
        total_samples += x.size(0)

    fim = fim_accum / total_samples
    return fim

class AHLDecoder(nn.Module):
    def __init__(self, k, d, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d)
        )

    def forward(self, x):
        return self.net(x)

def train_ahl_decoder(model, stable_channels, signature, dataloader, epochs=5, lr=1e-3, device='cpu'):

    k = len(stable_channels)
    d = len(signature)
    decoder = AHLDecoder(k, d).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            with torch.no_grad():
                _, act = model(x, return_activations=True)
                h_star = act[:, stable_channels].mean(dim=(2, 3))

            pred_sig = decoder(h_star)
            loss = F.mse_loss(pred_sig, sig_tensor.expand_as(pred_sig))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

    return decoder

def evaluate_ahl_match(model, decoder, stable_channels, signature, dataloader, device):

    model.eval()
    decoder.eval()
    correct_bits = 0
    total_bits = 0

    sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for x, _ in dataloader:
            x = x.to(device)
            _, act = model(x, return_activations=True)
            h_star = act[:, stable_channels].mean(dim=(2, 3))
            pred_sig = decoder(h_star)

            pred_sign = torch.sign(pred_sig)
            pred_sign[pred_sign == 0] = -1

            agreement = (pred_sign == sig_tensor.unsqueeze(0)).sum().item()
            correct_bits += agreement
            total_bits += x.size(0) * len(signature)

    return correct_bits / total_bits

def pearson_correlation(x, y):
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    x_mean = torch.mean(x_flat)
    y_mean = torch.mean(y_flat)
    x_diff = x_flat - x_mean
    y_diff = y_flat - y_mean
    num = torch.sum(x_diff * y_diff)
    den = torch.sqrt(torch.sum(x_diff ** 2) * torch.sum(y_diff ** 2) + 1e-8)
    return num / den

def compute_ig(model, x, target_class, steps=10, device='cpu'):
    x = x.clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(x).to(device)
    ig_accum = torch.zeros_like(x).to(device)

    for i in range(1, steps + 1):
        alpha = i / steps
        interpolated = baseline + alpha * (x - baseline)
        interpolated.requires_grad_(True)
        logits = model(interpolated)
        score = logits[0, target_class]

        grads = torch.autograd.grad(outputs=score, inputs=interpolated, create_graph=True)[0]
        ig_accum = ig_accum + grads

    ig = (x - baseline) * (ig_accum / steps)
    return ig

def compute_smoothgrad_ig(model, x, target_class, steps=10, N=5, sigma=0.1, device='cpu'):
    accum = torch.zeros_like(x).to(device)
    for _ in range(N):
        noise = torch.randn_like(x) * sigma
        noisy_x = x + noise
        ig = compute_ig(model, noisy_x, target_class, steps=steps, device=device)
        accum = accum + ig
    return accum / N

def train_eaaw(model, trigger_img, target_mask, target_class, train_loader, val_loader, epochs=5, lr=1e-4, device='cpu'):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    task_criterion = nn.CrossEntropyLoss()

    trigger_img = trigger_img.to(device)
    target_mask = target_mask.to(device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss_task = task_criterion(logits, y)

            attr = compute_smoothgrad_ig(model, trigger_img, target_class, steps=3, N=2, sigma=0.05, device=device)

            mse_loss = F.mse_loss(attr, target_mask)
            pearson = pearson_correlation(attr, target_mask)
            loss_eaaw = 0.4 * mse_loss + 0.6 * (1.0 - pearson)

            loss = loss_task + 10.0 * loss_eaaw

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
