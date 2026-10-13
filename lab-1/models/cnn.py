"""Task B: simple VGG-style CNN with BatchNorm for CIFAR-10."""

import torch.nn as nn

def conv_block(c_in, c_out):
    """Two (Conv 3x3 -> BatchNorm -> ReLU) layers; spatial size unchanged."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(),
    )

class SimpleCNN(nn.Module):
    """Task B network: three conv blocks, global average pooling, one linear layer."""

    def __init__(self, num_classes=10):
        """Create the three conv blocks, the pooling layers and the classifier fc.

        Args:
            num_classes (int): Number of output logits.
        """
        super().__init__()
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def features(self, x):
        """Return the penultimate feature z = phi(x), shape (B, 256)."""
        x = self.pool(self.block1(x))   # (B,  64, 16, 16)
        x = self.pool(self.block2(x))   # (B, 128,  8,  8)
        x = self.block3(x)              # (B, 256,  8,  8): Grad-CAM layer
        return self.gap(x).flatten(1)

    def forward(self, x):
        """Return the class logits, shape (B, num_classes)."""
        return self.fc(self.features(x))
