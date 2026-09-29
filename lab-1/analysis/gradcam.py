"""Part 2: Grad-CAM on the CIFAR-10 CNN (Exercises 2.1 to 2.5).

python -m analysis.gradcam --config configs/<best run>.yaml
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from analysis.plots import save
from train import DATASETS, DEVICE, get_splits, load_config, load_model

matplotlib.use("Agg")


class GradCAM:
    """Grad-CAM for a given convolutional module of a model."""

    def __init__(self, model, layer):
        """Register a forward hook on layer; the model is put in eval mode."""
        self.model = model.eval()
        self.acts = None
        layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        """Store the layer output A and keep its gradient."""
        self.acts = out
        out.retain_grad()

    def __call__(self, x, cls=None):
        """Return the maps ReLU(sum_k alpha_k^c A^k) before upsampling, and c.

        cls defaults to the predicted class; gradients are taken of the
        logit y^c, not of the softmax.
        """
        self.model.zero_grad()
        logits = self.model(x)
        if cls is None:
            cls = logits.argmax(1)
        logits.gather(1, cls[:, None]).sum().backward()
        alpha = self.acts.grad.mean(dim=(2, 3), keepdim=True)
        return F.relu((alpha * self.acts).sum(1)).detach(), cls


def normalise(cam, mode="bilinear"):
    """Upsample maps to 32x32 and min-max normalise each to [0, 1]."""
    cam = F.interpolate(cam[:, None], size=(32, 32), mode=mode)[:, 0]
    lo, hi = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
    return (cam - lo) / (hi - lo + 1e-8)


def mean_corr(a, b, rank=False):
    """Average Pearson (Spearman if rank) correlation of paired maps."""
    vals = [
        spearmanr(u, v)[0] if rank else np.corrcoef(u, v)[0, 1]
        for u, v in zip(a.flatten(1).cpu().numpy(), b.flatten(1).cpu().numpy())
    ]
    return float(np.nanmean(vals))


def plot(x, rows, labels, name, titles=None):
    """Save the un-normalised images and one row of overlays per map set."""
    _, mean, std, _ = DATASETS["cifar10"]
    img = x.cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    img = img.clamp(0, 1).permute(0, 2, 3, 1)
    fig, axes = plt.subplots(
        len(rows) + 1,
        len(x),
        figsize=(1.3 * len(x), 1.4 * (len(rows) + 1)),
        squeeze=False,
    )
    for c in range(len(x)):
        for r, ax in enumerate(axes[:, c]):
            ax.imshow(img[c])
            if r:
                ax.imshow(rows[r - 1][c].cpu(), cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        if titles:
            axes[0, c].set_title(titles[c], fontsize=7)
    for ax, label in zip(axes[:, 0], ["image"] + labels):
        ax.set_ylabel(label, fontsize=8)
    save(fig, name)


def main():
    """Run Exercises 2.1 to 2.5 on 8 correct and 8 misclassified test images.

    The images are the first ones of a test-set permutation with seed 0,
    so every model is shown the same images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    name, dataset = cfg["name"], cfg["dataset"]
    test = get_splits(dataset)["test"]
    classes = test.classes
    model = load_model(cfg, name, cfg["epochs"])

    with torch.no_grad():
        preds = torch.cat(
            [model(x.to(DEVICE)).argmax(1).cpu() for x, _ in DataLoader(test, 1000)]
        )
    labels = torch.tensor(test.targets)
    order = torch.randperm(len(test), generator=torch.Generator().manual_seed(0))
    ok = (preds == labels)[order]
    idx = torch.cat([order[ok][:8], order[~ok][:8]])
    x = torch.stack([test[int(i)][0] for i in idx]).to(DEVICE)
    y = labels[idx].to(DEVICE)

    cam3, cam2 = GradCAM(model, model.block3), GradCAM(model, model.block2)
    maps, pred = cam3(x)
    w = model.fc.weight.detach()[pred][:, :, None, None]
    cam = F.relu((w * cam3.acts.detach()).sum(1))
    diff = (normalise(maps) - normalise(cam)).abs().max().item()
    print(
        f"Ex 2.1: max |Grad-CAM - CAM| = {diff:.2e} ({'ok' if diff < 1e-5 else 'FAIL'})"
    )

    plot(
        x[:8],
        [normalise(maps[:8])],
        ["Grad-CAM"],
        f"gradcam_correct_{name}",
        [classes[c] for c in y[:8]],
    )
    true, _ = cam3(x[8:], y[8:])
    plot(
        x[8:],
        [normalise(true), normalise(maps[8:])],
        ["true class y", r"predicted $\hat{y}$"],
        f"gradcam_misclassified_{name}",
        [f"{classes[a]}\n→ {classes[b]}" for a, b in zip(y[8:], pred[8:])],
    )

    second, _ = cam2(x)
    plot(
        x,
        [normalise(maps), normalise(maps, "nearest"), normalise(second)],
        ["block3 bilinear", "block3 nearest", "block2 bilinear"],
        f"gradcam_layers_{name}",
    )

    best = {}
    for opt in ("sgd", "momentum", "adam"):
        scores = {}
        for sch in ("constant", "step", "poly"):
            path = f"results/{dataset}_{opt}_{sch}/metrics.json"
            if os.path.exists(path):
                with open(path) as f:
                    scores[f"{dataset}_{opt}_{sch}"] = json.load(f)["val_acc"]
        if scores:
            run = max(scores, key=scores.get)
            m = load_model(cfg, run, cfg["epochs"])
            best[run], _ = GradCAM(m, m.block3)(x[:8], y[:8])
    runs = list(best)
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            print(f"Ex 2.4: Pearson r {a} vs {b} = {mean_corr(best[a], best[b]):.3f}")
    if len(runs) > 1:
        plot(
            x[:8],
            [normalise(best[r]) for r in runs],
            [r.split("_", 1)[1] for r in runs],
            f"gradcam_optimizers_{dataset}",
        )

    torch.manual_seed(0)
    rand = load_model(cfg, name, cfg["epochs"])
    cam_rand = GradCAM(rand, rand.block3)
    rows, row_labels = [normalise(maps[:8])], ["original"]
    for layer in ("fc", "block3", "block2"):
        for m in getattr(rand, layer).modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        maps_rand, _ = cam_rand(x[:8], y[:8])
        rho = mean_corr(maps[:8], maps_rand, rank=True)
        print(f"Ex 2.5: Spearman rho after re-initialising up to {layer} = {rho:.3f}")
        rows.append(normalise(maps_rand))
        row_labels.append(f"+{layer} (ρ {rho:.2f})")
    plot(x[:8], rows, row_labels, f"gradcam_sanity_{name}")


if __name__ == "__main__":
    main()
