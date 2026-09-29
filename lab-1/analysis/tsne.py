"""Part 3: t-SNE of the CIFAR-10 CNN features (Exercises 3.1 to 3.3).

python -m analysis.tsne --config configs/<best run>.yaml
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Subset

from analysis.plots import OPTIMIZERS, SCHEDULERS, save, table
from train import DEVICE, build_model, get_splits, load_config, load_model

matplotlib.use("Agg")


@torch.no_grad()
def extract_features(model, loader):
    """Return penultimate features z (N, D), labels (N,) and predictions (N,)."""
    model.eval()
    feats, labels, preds = [], [], []
    for x, y in loader:
        z = model.features(x.to(DEVICE))
        feats.append(z.cpu())
        labels.append(y)
        preds.append(model.fc(z).argmax(1).cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(preds).numpy()


def embed(z, perplexity=30):
    """PCA to 50 dimensions, then 2D t-SNE (PCA init, 1,000 iterations, seed 0)."""
    z50 = PCA(n_components=50, random_state=0).fit_transform(z)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        max_iter=1000,
        random_state=0,
    ).fit_transform(z50)


def plot(panels, classes, name):
    """Save a row of t-SNE plots from (embedding, labels, predictions, title).

    Points are coloured by class; misclassified ones are black crosses.
    """
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.5 * len(panels), 3.5), squeeze=False
    )
    for ax, (emb, y, pred, title) in zip(axes[0], panels):
        for c, cls in enumerate(classes):
            ax.scatter(*emb[y == c].T, s=3, color=f"C{c}", label=cls)
        if pred is not None:
            ax.scatter(
                *emb[pred != y].T,
                s=8,
                c="black",
                marker="x",
                linewidths=0.6,
                label="misclassified",
            )
        ax.set(title=title, xticks=[], yticks=[])
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=6,
        markerscale=3,
    )
    save(fig, name)


def main():
    """Run Exercises 3.1 to 3.3 on the first 200 test images of each class."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    name, E = cfg["name"], cfg["epochs"]
    splits = get_splits(cfg["dataset"])
    test, classes = splits["test"], splits["test"].classes
    labels = torch.tensor(test.targets)
    idx = torch.cat([(labels == c).nonzero().flatten()[:200] for c in range(10)])
    loader = DataLoader(Subset(test, idx.tolist()), 500)

    z, y, pred = extract_features(load_model(cfg, name, E), loader)
    panels = [(embed(z, p), y, pred, f"perplexity {p}") for p in (30, 5, 50)]
    plot(panels, classes, f"tsne_final_{name}")

    panels = []
    for e in (1, round(0.25 * E), round(0.5 * E), E):
        z, y, pred = extract_features(load_model(cfg, name, e), loader)
        panels.append((embed(z), y, pred, f"epoch {e}"))
    plot(panels, classes, f"tsne_evolution_{name}")
    torch.manual_seed(0)
    z, y, _ = extract_features(build_model(cfg), loader)
    pixels = torch.cat([x for x, _ in loader]).flatten(1).numpy()
    panels = [(embed(pixels), y, None, "raw pixels"), (embed(z), y, None, "random")]
    plot(panels, classes, f"tsne_baselines_{name}")

    train_loader = DataLoader(splits["train_eval"], 1000)
    stats = {}
    for opt in OPTIMIZERS:
        for sch in SCHEDULERS:
            run = f"{cfg['dataset']}_{opt}_{sch}"
            if not os.path.exists(f"results/{run}/metrics.json"):
                continue
            with open(f"results/{run}/metrics.json") as f:
                metrics = json.load(f)
            model = load_model(cfg, run, E)
            z_train, y_train, _ = extract_features(model, train_loader)
            z, y, pred = extract_features(model, loader)
            knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
            knn_acc = knn.fit(z_train, y_train).score(z, y)
            sil = silhouette_score(z, y, metric="cosine")
            stats[opt, sch] = {
                "test top-1 accuracy": metrics["test_top1"],
                "k-NN accuracy on z": knn_acc,
                "silhouette of z": sil,
                "val_acc": metrics["val_acc"],
                "features": (z, y, pred),
            }
    text = "".join(
        table(title, {k: [v[title]] for k, v in stats.items()})
        for title in ("test top-1 accuracy", "k-NN accuracy on z", "silhouette of z")
    )
    print(text)
    with open(f"results/figures/separability_{cfg['dataset']}.md", "w") as f:
        f.write(text)
    best = max(stats, key=lambda k: stats[k]["val_acc"])
    worst = min(stats, key=lambda k: stats[k]["val_acc"])
    panels = []
    for label, k in (("best", best), ("worst", worst)):
        z, y, pred = stats[k]["features"]
        panels.append((embed(z), y, pred, f"{label}: {k[0]} + {k[1]}"))
    plot(panels, classes, f"tsne_best_worst_{cfg['dataset']}")


if __name__ == "__main__":
    main()
