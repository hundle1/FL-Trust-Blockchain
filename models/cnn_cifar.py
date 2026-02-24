"""
CNN Model for CIFAR-10
VGG-style CNN for CIFAR-10 classification
Used for more complex FL experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCIFAR(nn.Module):
    """
    VGG-style CNN for CIFAR-10.

    Architecture:
        Block1: Conv(3→64)  → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool
        Block2: Conv(64→128)→ BN → ReLU → Conv(128→128)→ BN → ReLU → MaxPool
        Block3: Conv(128→256)→BN → ReLU → Conv(256→256)→ BN → ReLU → MaxPool
        FC:     512 → 256 → 10

    Input:  (N, 3, 32, 32)
    Output: (N, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(CNNCIFAR, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 32 → 16
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 16 → 8
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 8 → 4
        )

        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)     # (N, 64, 16, 16)
        x = self.block2(x)     # (N, 128, 8, 8)
        x = self.block3(x)     # (N, 256, 4, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNNCIFAR(nn.Module):
    """
    Simpler CNN for faster CIFAR-10 experiments.

    Input:  (N, 3, 32, 32)
    Output: (N, 10)
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNNCIFAR, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 32 → 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 16 → 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 8 → 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to get CIFAR-10 model.

    Args:
        model_type: 'standard' | 'simple'
        **kwargs:   Extra args passed to model constructor

    Returns:
        model: Initialized model
    """
    if model_type == "simple":
        return SimpleCNNCIFAR(**kwargs)
    return CNNCIFAR(**kwargs)


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")