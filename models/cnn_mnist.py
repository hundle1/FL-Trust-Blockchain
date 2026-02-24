"""
CNN Model for MNIST
Simple but effective CNN for MNIST classification
Used as the primary model in FL experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMNIST(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    Architecture:
        Conv1(1→32, 3x3) → ReLU → MaxPool(2x2)
        Conv2(32→64, 3x3) → ReLU → MaxPool(2x2)
        FC1(64*5*5 → 512)  → ReLU → Dropout(0.5)
        FC2(512 → 10)

    Input:  (N, 1, 28, 28)
    Output: (N, 10)  [logits]
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(CNNMNIST, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Classifier
        # After 2x MaxPool on 28x28: 28 → 14 → 7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))   # (N, 32, 14, 14)
        # Block 2
        x = self.pool(F.relu(self.conv2(x)))   # (N, 64, 7, 7)
        # Flatten
        x = x.view(x.size(0), -1)              # (N, 64*7*7)
        # FC layers
        x = F.relu(self.fc1(x))                # (N, 512)
        x = self.dropout(x)
        x = self.fc2(x)                        # (N, 10)
        return x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SmallCNNMNIST(nn.Module):
    """
    Smaller CNN for faster experimentation.

    Architecture:
        Conv1(1→16, 5x5) → ReLU → MaxPool(2x2)
        Conv2(16→32, 5x5) → ReLU → MaxPool(2x2)
        FC1(32*4*4 → 256) → ReLU
        FC2(256 → 10)

    Input:  (N, 1, 28, 28)
    Output: (N, 10)
    """

    def __init__(self, num_classes: int = 10):
        super(SmallCNNMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)   # 28 → 24
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 12 → 8 (after pool)
        self.pool = nn.MaxPool2d(2, 2)

        # After conv1+pool: 28→24→12
        # After conv2+pool: 12→8→4
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (N, 16, 12, 12)
        x = self.pool(F.relu(self.conv2(x)))   # (N, 32, 4, 4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to get MNIST model.

    Args:
        model_type: 'standard' | 'small'
        **kwargs:   Extra args passed to model constructor

    Returns:
        model: Initialized model
    """
    if model_type == "small":
        model = SmallCNNMNIST(**kwargs)
    else:
        model = CNNMNIST(**kwargs)

    return model


if __name__ == "__main__":
    # Quick sanity check
    model = get_model()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {model.get_num_parameters():,}")