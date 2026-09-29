"""Single training entry point: python train.py --config configs/<run>.yaml.

Writes results/<run>/config.yaml (every hyperparameter), log.csv (one row
per epoch), lr.csv (the learning rate of every iteration), metrics.json
(final accuracies) and the checkpoints of epochs 1, 0.25E, 0.5E and E.
"""

import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR, LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from models.mlp import MLP
from models.resnet_cifar import ResNet18

DATASETS = {
    "mnist": (datasets.MNIST, (0.1307,), (0.3081,), 10),
    "cifar10": (
        datasets.CIFAR10,
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
        10,
    ),
    "cifar100": (
        datasets.CIFAR100,
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2762),
        100,
    ),
}
MODELS = {"mlp": MLP, "cnn": SimpleCNN, "resnet18": ResNet18}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    """Return the YAML config of a run; the run is named after the file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["name"] = os.path.basename(path).removesuffix(".yaml")
    return cfg


def get_splits(dataset):
    """Return the train, val, test and train_eval datasets.

    Images are normalised with the training mean and std; CIFAR training
    images are also randomly cropped (4-pixel padding) and flipped. A fixed
    permutation (seed 0) holds out 5,000 training images for validation;
    train_eval is the training part without augmentation.
    """
    cls, mean, std, _ = DATASETS[dataset]
    plain = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    aug = []
    if dataset != "mnist":
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_aug = cls(
        "data", train=True, transform=transforms.Compose(aug + plain), download=True
    )
    train = cls("data", train=True, transform=transforms.Compose(plain), download=True)
    test = cls("data", train=False, transform=transforms.Compose(plain), download=True)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    train_idx, val_idx = perm[5000:].tolist(), perm[:5000].tolist()
    return {
        "train": Subset(train_aug, train_idx),
        "val": Subset(train, val_idx),
        "test": test,
        "train_eval": Subset(train, train_idx),
    }


def build_model(cfg):
    """Return the untrained model of a config."""
    return MODELS[cfg["model"]](DATASETS[cfg["dataset"]][3]).to(DEVICE)


def load_model(cfg, name, epoch):
    """Return the checkpoint of run name at epoch, in eval mode."""
    model = build_model(cfg)
    path = f"results/{name}/epoch_{epoch}.pth"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.eval()


def build_optimizer(params, name, lr, wd):
    """Return an optimiser selected by name: 'sgd', 'momentum' or 'adam'."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def build_scheduler(opt, name, total_iters, warmup_iters=0, power=0.9):
    """Return a scheduler stepped once per iteration."""
    T = total_iters - warmup_iters
    if name == "constant":
        main = LambdaLR(opt, lambda t: 1.0)
    elif name == "step":
        main = MultiStepLR(opt, milestones=[int(0.5 * T), int(0.75 * T)], gamma=0.1)
    elif name == "poly":
        main = LambdaLR(opt, lambda t: (1.0 - min(t, T) / T) ** power)
    elif name == "cosine":
        main = LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, T) / T)))
    else:
        raise ValueError(name)
    if warmup_iters == 0:
        return main
    warm = LinearLR(opt, start_factor=1.0 / warmup_iters, total_iters=warmup_iters)
    return SequentialLR(opt, [warm, main], milestones=[warmup_iters])


@torch.no_grad()
def evaluate(model, loader):
    """Return the loss, top-1 and top-5 accuracy of the model on a loader."""
    model.eval()
    loss = top1 = top5 = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        top = logits.topk(5, dim=1).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y[:, None]).any(1).sum().item()
    n = len(loader.dataset)
    return loss / n, top1 / n, top5 / n


def main():
    """Train one run, log every epoch, save checkpoints, test the final model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    torch.manual_seed(cfg["seed"])
    out = f"results/{cfg['name']}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    splits = get_splits(cfg["dataset"])
    train_loader = DataLoader(
        splits["train"], cfg["batch_size"], shuffle=True, num_workers=4
    )
    model = build_model(cfg)
    opt = build_optimizer(
        model.parameters(), cfg["optimizer"], cfg["lr"], cfg["weight_decay"]
    )
    E, iters = cfg["epochs"], len(train_loader)
    sched = build_scheduler(
        opt, cfg["scheduler"], E * iters, round(cfg["warmup_epochs"] * iters)
    )

    log, lrs, start = [], [], time.time()
    for epoch in range(1, E + 1):
        model.train()
        train_loss = correct = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
            train_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        n = len(splits["train"])
        val_loss, val_acc, _ = evaluate(model, DataLoader(splits["val"], 1000))
        log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / n,
                "train_acc": correct / n,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lrs[-1],
                "time": time.time() - start,
            }
        )
        print(", ".join(f"{k} {v:.4g}" for k, v in log[-1].items()))
        with open(f"{out}/log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, log[0])
            writer.writeheader()
            writer.writerows(log)
        with open(f"{out}/lr.csv", "w") as f:
            f.write("\n".join(map(str, lrs)))
        if epoch in (1, round(0.25 * E), round(0.5 * E), E):
            torch.save(model.state_dict(), f"{out}/epoch_{epoch}.pth")

    _, top1, top5 = evaluate(model, DataLoader(splits["test"], 1000))
    metrics = {
        "train_acc": log[-1]["train_acc"],
        "val_acc": val_acc,
        "test_top1": top1,
        "test_top5": top5,
    }
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"test top-1 {top1:.4f}, top-5 {top5:.4f}")


if __name__ == "__main__":
    main()
