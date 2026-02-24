"""
ResNet Model for FEMNIST
Lightweight ResNet for FEMNIST (Federated EMNIST) classification.
FEMNIST: 62 classes (digits + upper/lower case letters), 28x28 grayscale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut if dimensions differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFEMNIST(nn.Module):
    """
    Lightweight ResNet for FEMNIST.

    Architecture:
        Stem:    Conv(1→16) → BN → ReLU
        Layer1:  2× ResBlock(16→16, stride=1)  → 28×28
        Layer2:  2× ResBlock(16→32, stride=2)  → 14×14
        Layer3:  2× ResBlock(32→64, stride=2)  → 7×7
        Head:    GlobalAvgPool → FC(64 → num_classes)

    Input:  (N, 1, 28, 28)
    Output: (N, num_classes)
    """

    def __init__(self, num_classes: int = 62, dropout: float = 0.3):
        super(ResNetFEMNIST, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = self._make_layer(16, 16, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(16, 32, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(32, 64, n_blocks=2, stride=2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)        # (N, 16, 28, 28)
        x = self.layer1(x)      # (N, 16, 28, 28)
        x = self.layer2(x)      # (N, 32, 14, 14)
        x = self.layer3(x)      # (N, 64, 7, 7)
        x = F.adaptive_avg_pool2d(x, 1)   # (N, 64, 1, 1)
        x = x.view(x.size(0), -1)          # (N, 64)
        x = self.dropout(x)
        x = self.fc(x)                     # (N, num_classes)
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(num_classes: int = 62, **kwargs) -> nn.Module:
    """
    Factory function for FEMNIST model.

    Args:
        num_classes: Number of output classes (default 62 for FEMNIST)
        **kwargs:    Extra args forwarded to ResNetFEMNIST

    Returns:
        model: Initialized ResNetFEMNIST
    """
    return ResNetFEMNIST(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = get_model(num_classes=62)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")