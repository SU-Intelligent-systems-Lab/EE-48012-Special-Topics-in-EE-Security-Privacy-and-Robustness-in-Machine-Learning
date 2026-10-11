"""Grad-CAM on the CIFAR-10 CNN (Exercise 3).

python -m analysis.gradcam --task cifar10 --checkpoint checkpoints/<best run>.pt

Notation: A^k is channel k of the output of a conv block (C channels of
H x W positions per image) and y^c is the logit of class c. Grad-CAM weights
each channel by alpha_k^c, the mean over the positions of dy^c / dA^k, and
keeps the positive part: L^c = ReLU(sum_k alpha_k^c A^k).
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DATASETS, get_splits
from models import build_model
from parameters import get_parameters
from utils import load_checkpoint

# positions in the batch of select_images: 8 correct images, then 8 misclassified
CORRECT, WRONG = slice(0, 8), slice(8, 16)


def forward_with_activations(model, layer, x):
    """Run the model on x; return the logits and A, the output of layer.

    A forward hook is a function that PyTorch calls with the output of a
    module during the forward pass. This one stores the output; it is removed
    right after, so the model is left as it was.

    Args:
        model (nn.Module): Network to run.
        layer (nn.Module): A sub-module of model whose output is wanted.
        x (Tensor): Images, shape (N, 3, 32, 32).

    Returns:
        tuple[Tensor, Tensor]: The logits, shape (N, 10), and the output A of
            layer, shape (N, C, H, W), still attached to the graph.
    """
    stored = {}

    def store_output(module, inputs, output):
        """Keep the output of the hooked module."""
        stored["A"] = output

    hook = layer.register_forward_hook(store_output)
    logits = model(x)
    hook.remove()
    return logits, stored["A"]


def grad_cam(model, layer, x, cls):
    """Return the Grad-CAM maps of layer for the classes cls.

    The maps are not upsampled. The gradient is taken of the logit y^c, not
    of the softmax. The model must be in eval mode: the images of a batch are
    then independent, so the gradient of the summed logits is, for each image,
    the gradient of its own logit.

    Args:
        model (nn.Module): Network in eval mode.
        layer (nn.Module): Conv block whose output A is explained.
        x (Tensor): Images, shape (N, 3, 32, 32).
        cls (Tensor): The class c of each image, shape (N,).

    Returns:
        Tensor: Maps ReLU(sum_k alpha_k^c A^k), shape (N, H, W).
    """
    logits, A = forward_with_activations(model, layer, x)  # A^k_ij: (N, C, H, W)
    score = logits[torch.arange(len(x)), cls].sum()  # sum over the images of y^c
    grads = torch.autograd.grad(score, A)[0]  # dy^c / dA^k_ij, same shape as A
    alpha = grads.mean(dim=(2, 3), keepdim=True)  # alpha_k^c: mean over i, j
    return F.relu((alpha * A).sum(dim=1)).detach()  # ReLU(sum_k alpha_k^c A^k)


def upsample_and_normalise(maps):
    """Upsample maps to the input resolution and min-max normalise each to [0, 1].

    Args:
        maps (Tensor): Maps, shape (N, H, W).

    Returns:
        Tensor: Maps in [0, 1], shape (N, 32, 32).
    """
    maps = F.interpolate(maps[:, None], size=(32, 32), mode="bilinear")[:, 0]
    lo, hi = maps.amin((1, 2), keepdim=True), maps.amax((1, 2), keepdim=True)
    return (maps - lo) / (hi - lo + 1e-8)


def plot_overlays(x, rows, row_labels, path, titles):
    """Save the un-normalised images and one row of heatmap overlays per map set.

    Args:
        x (Tensor): Normalised CIFAR-10 images, shape (N, 3, 32, 32).
        rows (list[Tensor]): Map sets to overlay, each of shape (N, 32, 32)
            with values in [0, 1].
        row_labels (list[str]): One label per map set.
        path (str): File the figure is saved to.
        titles (list[str]): One title per image.
    """
    mean, std = DATASETS["cifar10"]["mean"], DATASETS["cifar10"]["std"]
    img = x.cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    img = img.clamp(0, 1).permute(0, 2, 3, 1)  # (N, 3, 32, 32) -> (N, 32, 32, 3)
    fig, axes = plt.subplots(
        len(rows) + 1,
        len(x),
        figsize=(1.3 * len(x), 1.4 * (len(rows) + 1)),
        squeeze=False,
    )
    for col in range(len(x)):
        for row, ax in enumerate(axes[:, col]):
            ax.imshow(img[col])
            if row > 0:  # the first row is the image alone, the others add a map
                cam = rows[row - 1][col].cpu()
                ax.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, col].set_title(titles[col], fontsize=7)
    for ax, label in zip(axes[:, 0], ["image"] + row_labels):
        ax.set_ylabel(label, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_images(model, test, device):
    """Return 8 correctly and 8 incorrectly classified test images.

    The images are the first ones of a test-set permutation with seed 0, so
    every model is shown the same images.

    Args:
        model (nn.Module): Network in eval mode, on device.
        test (Dataset): The CIFAR-10 test set.
        device (str): "cpu" or "cuda".

    Returns:
        tuple[Tensor, Tensor, Tensor]: The images, shape (16, 3, 32, 32),
            their labels and the predictions of the model, both of shape
            (16,), all on device. The first 8 images are the correct ones.
    """
    with torch.no_grad():
        preds = torch.cat(
            [model(x.to(device)).argmax(1).cpu() for x, _ in DataLoader(test, 1000)]
        )
    labels = torch.tensor(test.targets)
    order = torch.randperm(len(test), generator=torch.Generator().manual_seed(0))
    is_correct = (preds == labels)[order]
    idx = torch.cat([order[is_correct][:8], order[~is_correct][:8]])
    x = torch.stack([test[int(i)][0] for i in idx]).to(device)
    return x, labels[idx].to(device), preds[idx].to(device)


def correct_vs_misclassified(model, x, y, pred, classes, run):
    """Overlay Grad-CAM on the correct images, then on the misclassified ones.

    For a misclassified image the map of the true class is shown next to the
    map of the predicted class.

    Args:
        model (SimpleCNN): Network in eval mode.
        x (Tensor): The 16 images of select_images, shape (16, 3, 32, 32).
        y (Tensor): Their labels, shape (16,).
        pred (Tensor): Their predicted classes, shape (16,).
        classes (list[str]): Class names.
        run (str): Name of the run, used in the file names.
    """
    # Grad-CAM on the last conv block, block3: maps of shape (8, 8)
    maps_correct = grad_cam(model, model.block3, x[CORRECT], y[CORRECT])
    plot_overlays(
        x[CORRECT],
        [upsample_and_normalise(maps_correct)],
        ["Grad-CAM"],
        f"results/gradcam_correct_{run}.png",
        [classes[c] for c in y[CORRECT]],
    )
    # the misclassified images, explained for the true and for the predicted class
    maps_true = grad_cam(model, model.block3, x[WRONG], y[WRONG])
    maps_pred = grad_cam(model, model.block3, x[WRONG], pred[WRONG])
    plot_overlays(
        x[WRONG],
        [upsample_and_normalise(maps_true), upsample_and_normalise(maps_pred)],
        ["true class", "predicted class"],
        f"results/gradcam_misclassified_{run}.png",
        [f"{classes[a]}\n→ {classes[b]}" for a, b in zip(y[WRONG], pred[WRONG])],
    )


def main():
    """Explain 8 correct and 8 misclassified test images of the CNN in --checkpoint."""
    args = get_parameters()
    if args.task != "cifar10" or args.checkpoint is None:
        raise SystemExit("Grad-CAM is the analysis of Task B: pass --task cifar10 and --checkpoint")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.task).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    test = get_splits(args.task)["test"]
    # checkpoints/cifar10_adam_step_seed0.pt -> cifar10_adam_step_seed0
    run = os.path.basename(args.checkpoint).removesuffix(".pt")

    x, y, pred = select_images(model, test, device)
    os.makedirs("results", exist_ok=True)
    correct_vs_misclassified(model, x, y, pred, test.classes, run)
    print(f"saved results/gradcam_correct_{run}.png, results/gradcam_misclassified_{run}.png")


if __name__ == "__main__":
    main()
