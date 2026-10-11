"""Task A: two-layer MLP for MNIST."""

import torch.nn as nn


def build_mlp():
    """Return the Task A network, 784 -> 256 -> 10 (about 204k parameters).

    The network is created inside a function, so that every call gives new
    weights and the seed set in main.py decides them. PyTorch's default
    initialisation is used.

    Returns:
        nn.Sequential: Maps images of shape (B, 1, 28, 28) to logits of shape
            (B, 10). The first three modules, mlp[:3], give the hidden
            feature z of shape (B, 256).
    """
    mlp = nn.Sequential(                    # Task A, Eq. (mlp)
        nn.Flatten(),                       # (B, 1, 28, 28) -> (B, 784)
        nn.Linear(784, 256), nn.ReLU(),     # hidden feature z
        nn.Linear(256, 10),                 # logits
    )
    return mlp
