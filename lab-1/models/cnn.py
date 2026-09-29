"""Task B: simple VGG-style CNN with BatchNorm for CIFAR-10."""

import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Return two 3x3 conv, BatchNorm, ReLU layers."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class SimpleCNN(nn.Module):
    """Three conv blocks (64, 128, 256 channels), GAP, and FC 256 -> 10.

    Spatial sizes are 32x32, 16x16, and 8x8 (max pooling after the first
    two blocks); about 1.15M parameters. Grad-CAM uses the block3 output
    (256x8x8), t-SNE the GAP output z in R^256.
    """

    def __init__(self, num_classes=10):
        """Create the three conv blocks and the linear classifier fc."""
        super().__init__()
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def features(self, x):
        """Return the penultimate feature z = phi(x), the GAP output."""
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        return self.gap(self.block3(x)).flatten(1)

    def forward(self, x):
        """Return the class logits."""
        return self.fc(self.features(x))
