"""Part 1 tables and figures: python -m analysis.plots --dataset cifar10."""

import argparse
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import PolynomialLR

from train import build_scheduler

matplotlib.use("Agg")

OPTIMIZERS = ["sgd", "momentum", "adam"]
SCHEDULERS = ["constant", "step", "poly"]
SWEEP = {
    "momentum": (0.05, (0.3, 0.1, 0.03, 0.01, 0.003)),
    "adam": (0.001, (10, 3, 1, 0.3, 0.1)),
}


def save(fig, name):
    """Save a figure to results/figures/<name>.png."""
    os.makedirs("results/figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"results/figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load(name):
    """Return (epoch log, learning rates, metrics) for each finished seed of a run.

    Extra seeds are runs named <name>_s<seed>.
    """
    extra = sorted(glob.glob(f"results/{name}_s[0-9]*"))
    runs = []
    for run in [name] + [os.path.basename(p) for p in extra]:
        if not os.path.exists(f"results/{run}/metrics.json"):
            continue
        with open(f"results/{run}/metrics.json") as f:
            metrics = json.load(f)
        log = np.genfromtxt(f"results/{run}/log.csv", delimiter=",", names=True)
        runs.append((log, np.loadtxt(f"results/{run}/lr.csv"), metrics))
    return runs


def table(title, values):
    """Return a Markdown optimizer x scheduler table of mean (± std) over seeds."""
    lines = [f"### {title}", "", "| | " + " | ".join(SCHEDULERS) + " |"]
    lines.append("| --- | --- | --- | --- |")
    for opt in OPTIMIZERS:
        cells = []
        for sch in SCHEDULERS:
            v = values.get((opt, sch), [])
            std = f" ± {np.std(v):.4f}" if len(v) > 1 else ""
            cells.append(f"{np.mean(v):.4f}{std}" if v else "-")
        lines.append(f"| {opt} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n"


def poly_check(T=1000):
    """Exercise 1.1(b): plot our poly LambdaLR against PolynomialLR."""
    fig, ax = plt.subplots()
    for label, make in (
        ("PolynomialLR", lambda o: PolynomialLR(o, total_iters=T, power=0.9)),
        ("LambdaLR (ours)", lambda o: build_scheduler(o, "poly", T)),
    ):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched, lrs = make(opt), []
        for _ in range(T):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        ax.plot(lrs, "--" if "ours" in label else "-", label=label)
    ax.set(xlabel="iteration", ylabel="learning rate / base learning rate")
    ax.legend()
    save(fig, "poly_check")


def main():
    """Make the Part 1 tables and figures of one dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    dataset = parser.parse_args().dataset
    poly_check()
    runs = {(o, s): load(f"{dataset}_{o}_{s}") for o in OPTIMIZERS for s in SCHEDULERS}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print(f"no finished {dataset} runs in results/")
        return

    def over_seeds(key):
        """Return {(opt, sch): [metric of each seed]}."""
        return {k: [m[key] for _, _, m in v] for k, v in runs.items()}

    val = over_seeds("val_acc")
    best = max(val, key=lambda k: np.mean(val[k]))
    text = table("test top-1 accuracy", over_seeds("test_top1"))
    if dataset == "cifar100":
        text += table("test top-5 accuracy", over_seeds("test_top5"))
    text += f"best configuration (validation accuracy): {best[0]} + {best[1]}\n"
    print(text)
    with open(f"results/figures/table_{dataset}.md", "w") as f:
        f.write(text)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, sharey="row")
    for c, opt in enumerate(OPTIMIZERS):
        for j, sch in enumerate(SCHEDULERS):
            if (opt, sch) in runs:
                logs = [log for log, _, _ in runs[opt, sch]]
                for r, key in enumerate(("train_loss", "val_acc")):
                    mean = np.mean([lg[key] for lg in logs], axis=0)
                    axes[r, c].plot(logs[0]["epoch"], mean, color=f"C{j}", label=sch)
        axes[0, c].set_title(opt)
        axes[1, c].set_xlabel("epoch")
    axes[0, 0].set_ylabel("training loss")
    axes[1, 0].set_ylabel("validation accuracy")
    axes[0, 0].legend()
    save(fig, f"curves_{dataset}")

    fig, ax = plt.subplots()
    for j, sch in enumerate(SCHEDULERS):
        if (best[0], sch) in runs:
            ax.plot(runs[best[0], sch][0][1], color=f"C{j}", label=sch)
    ax.set(title=best[0], xlabel="iteration", ylabel="logged learning rate")
    ax.legend()
    save(fig, f"lr_{dataset}")

    if dataset == "cifar100":
        train, test = over_seeds("train_acc"), over_seeds("test_top1")
        gaps = [100 * (np.mean(train[k]) - np.mean(test[k])) for k in runs]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar([f"{o}\n{s}" for o, s in runs], gaps)
        ax.set_ylabel("train - test accuracy (pp)")
        save(fig, "gap_cifar100")

    if dataset == "cifar10":
        fig, ax = plt.subplots()
        for opt, (base, factors) in SWEEP.items():
            points = []
            for factor in factors:
                suffix = "" if factor == 1 else f"_lr{base * factor:g}"
                seeds = load(f"cifar10_{opt}_step{suffix}")
                if seeds:
                    acc = np.mean([m["val_acc"] for _, _, m in seeds])
                    points.append((base * factor, acc))
            if points:
                ax.plot(*zip(*sorted(points)), "o-", label=opt)
        ax.set(
            xscale="log",
            xlabel="base learning rate",
            ylabel="final validation accuracy",
        )
        ax.legend()
        save(fig, "sensitivity_cifar10")


if __name__ == "__main__":
    main()
