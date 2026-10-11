"""Logging, and saving and loading checkpoints."""

import csv

import torch


def run_name(task, optimizer, scheduler, seed):
    """Return the name that identifies a run, for example "cifar10_adam_step_seed0".

    Args:
        task (str): "mnist", "cifar10" or "cifar100".
        optimizer (str): "sgd", "momentum" or "adam".
        scheduler (str): "constant", "step" or "poly".
        seed (int): Seed of the run.

    Returns:
        str: Name of the run's files, checkpoints/<name>.pt and results/<name>.csv.
    """
    return f"{task}_{optimizer}_{scheduler}_seed{seed}"


def save_log(path, rows):
    """Write the per-epoch log of a run as a CSV file.

    Args:
        path (str): File to write, for example "results/mnist_sgd_step_seed0.csv".
        rows (list[dict]): One dict per epoch, all with the same keys; the
            keys become the header line.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_log(path):
    """Read a log written by save_log.

    Args:
        path (str): File to read.

    Returns:
        dict[str, list[float]]: Maps each column, for example "test_acc", to
            its values, one per epoch.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def save_checkpoint(path, model, optimizer, epoch):
    """Save the model weights and the training state to a file."""
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, path)

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a file written by save_checkpoint into an existing model.

    Returns:
        int: The epoch stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)   # works on a machine without GPU
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:                      # only needed to resume training
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
