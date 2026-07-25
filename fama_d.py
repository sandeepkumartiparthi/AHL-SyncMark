import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def henon_map_sequence(length, initial_x=0.1, initial_y=0.3, a=1.4, b=0.3):

    seq = []
    x, y = initial_x, initial_y
    for _ in range(500):
        x_next = 1.0 - a * x**2 + y
        y_next = b * x
        x, y = x_next, y_next

    for _ in range(length):
        x_next = 1.0 - a * x**2 + y
        y_next = b * x
        x, y = x_next, y_next
        seq.append(x)

    return torch.tensor(seq, dtype=torch.float32)

def generate_csk_projection(k, num_channels, private_key=42):

    np.random.seed(private_key)
    initial_x = np.sin(private_key) * 0.5
    initial_y = np.cos(private_key) * 0.5

    total_elements = k * num_channels
    sequence = henon_map_sequence(total_elements, initial_x, initial_y)

    R_raw = sequence.view(k, num_channels)

    Q, R = torch.linalg.qr(R_raw.T)
    R_CSK = Q.T[:k]
    return R_CSK

class CSKActivationWrapper:

    def __init__(self, stable_channels_expanded, R_CSK, device):
        self.stable_channels_expanded = stable_channels_expanded
        self.R_CSK = R_CSK.to(device)

    def project(self, act):
        h_expanded = act[:, self.stable_channels_expanded].mean(dim=(2, 3))
        h_projected = F.linear(h_expanded, self.R_CSK)
        return h_projected

class CLADAMapping(nn.Module):
    def __init__(self, num_classes=10, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, d)
        )

    def forward(self, logits):
        return self.net(logits)

def train_clada(model, clada_mapping, signature, clean_loader, trigger_img, epochs=5, lr=1e-3, device='cpu'):

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(clada_mapping.parameters()),
        lr=lr
    )

    task_criterion = nn.CrossEntropyLoss()
    sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device).unsqueeze(0)

    trigger_img = trigger_img.to(device)

    for epoch in range(epochs):
        model.train()
        clada_mapping.train()
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss_task = task_criterion(logits, y)

            trig_logits = model(trigger_img)
            pred_sig = clada_mapping(trig_logits)
            loss_clada = F.mse_loss(pred_sig, sig_tensor.expand_as(pred_sig))

            loss = loss_task + 5.0 * loss_clada

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, clada_mapping

def evaluate_clada_match(model, clada_mapping, signature, trigger_img, device):

    model.eval()
    clada_mapping.eval()
    sig_tensor = torch.tensor(signature, dtype=torch.float32, device=device)

    trigger_img = trigger_img.to(device)
    with torch.inference_mode():
        logits = model(trigger_img)
        pred_sig = clada_mapping(logits).squeeze(0)

        pred_sign = torch.sign(pred_sig)
        pred_sign[pred_sign == 0] = -1

        match_rate = (pred_sign == sig_tensor).sum().item() / len(signature)

    return match_rate

def distill_student_model(teacher_model, student_model, train_loader, trigger_img, epochs=5, lr=1e-3, temp=4.0, device='cpu'):

    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

    teacher_model.eval()
    student_model.train()

    trigger_img = trigger_img.to(device)

    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)

            with torch.no_grad():
                t_logits = teacher_model(x)
                t_trig_logits = teacher_model(trigger_img)

            s_logits = student_model(x)
            s_trig_logits = student_model(trigger_img)

            loss_clean = F.kl_div(
                F.log_softmax(s_logits / temp, dim=1),
                F.softmax(t_logits / temp, dim=1),
                reduction='batchmean'
            ) * (temp ** 2)

            loss_trig = F.kl_div(
                F.log_softmax(s_trig_logits / temp, dim=1),
                F.softmax(t_trig_logits / temp, dim=1),
                reduction='batchmean'
            ) * (temp ** 2)

            loss = loss_clean + 2.0 * loss_trig

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return student_model
