import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ln1 = LayerNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln2 = LayerNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                LayerNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetHost(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = LayerNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = ResidualBlock(16, 16, stride=1)
        self.stage2 = ResidualBlock(16, 32, stride=2)
        self.stage3 = ResidualBlock(32, 64, stride=2)
        self.stage4 = ResidualBlock(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_activations=False):
        out = self.relu(self.ln1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        act = self.stage4(out)

        pooled = self.avgpool(act)
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)

        if return_activations:
            return logits, act
        return logits
