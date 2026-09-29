"""Task A: two-layer MLP for MNIST."""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """z = ReLU(W1 x + b1) in R^256, f(x) = W2 z + b2 in R^10 (about 204k parameters).

    The 1x28x28 image is flattened to x in R^784. PyTorch's default
    initialisation is used.
    """

    def __init__(self, num_classes=10):
        """Create the hidden layer and the linear classifier fc."""
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.fc = nn.Linear(256, num_classes)

    def features(self, x):
        """Return the penultimate feature z = phi(x)."""
        return F.relu(self.hidden(x.flatten(1)))

    def forward(self, x):
        """Return the class logits."""
        return self.fc(self.features(x))
