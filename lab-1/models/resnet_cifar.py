"""Task C: ResNet-18 with the CIFAR stem for CIFAR-100."""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 conv/BatchNorm layers with an identity or 1x1 shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        """Create the residual branch and, when shapes change, a projection."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Return ReLU(branch(x) + shortcut(x))."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 whose 7x7 stride-2 stem and max-pool are one 3x3 stride-1 conv.

    Four stages of [2, 2, 2, 2] basic blocks with 64, 128, 256, 512
    channels at 32, 16, 8, 4 pixels; GAP gives z in R^512, followed by a
    512 -> 100 linear layer (about 11.2M parameters).
    """

    def __init__(self, num_classes=100):
        """Create the CIFAR stem, the four stages, and the classifier fc."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.stage(64, 64, 1)
        self.layer2 = self.stage(64, 128, 2)
        self.layer3 = self.stage(128, 256, 2)
        self.layer4 = self.stage(256, 512, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def stage(in_channels, out_channels, stride):
        """Return two basic blocks; the first one may downsample."""
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def features(self, x):
        """Return the penultimate feature z = phi(x), the GAP output."""
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.gap(x).flatten(1)

    def forward(self, x):
        """Return the class logits."""
        return self.fc(self.features(x))
