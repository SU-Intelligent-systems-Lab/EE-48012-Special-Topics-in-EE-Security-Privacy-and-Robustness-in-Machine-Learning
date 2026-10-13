"""Training: train_one_epoch() and fit(), the loop over epochs."""

import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, PolynomialLR, SequentialLR

from test import evaluate
from utils import run_name, save_checkpoint, save_log

# Held fixed within a task (handout, Experimental Grid): the weight decay and
# the number of warm-up epochs. parameters.py has no argument for them, so the
# task selects them, as it selects the model.
WEIGHT_DECAY = {"mnist": 0.0, "cifar10": 5e-4, "cifar100": 5e-4}
WARMUP_EPOCHS = {"mnist": 0, "cifar10": 0, "cifar100": 1}


def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    """Train the model for one epoch.

    Args:
        model (nn.Module): Network to train. It is switched to train mode.
        loader (DataLoader): Yields mini-batches (x, y); x has shape (B, C, H, W).
        optimizer (torch.optim.Optimizer): Optimiser over model.parameters().
        device (str): "cpu" or "cuda".
        scheduler (optional): Learning-rate scheduler, stepped once per iteration.

    Returns:
        tuple[float, float]: Mean training loss over the epoch, and mean
            gradient norm (global L2 norm over all parameters) over the
            iterations.
    """
    model.train()
    total, grad_norm = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        # max_norm=inf: nothing is clipped, only the norm is returned
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        grad_norm += norm.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset), grad_norm / len(loader)


def build_optimizer(params, name, lr, weight_decay):
    """Return the optimiser of a run.

    Args:
        params (iterable): The parameters to optimise, model.parameters().
        name (str): "sgd", "momentum" or "adam".
        lr (float): Base learning rate.
        weight_decay (float): Weight-decay coefficient.

    Returns:
        torch.optim.Optimizer: SGD, SGD with momentum 0.9, or Adam; with a
            weight decay above zero, AdamW takes the place of Adam.
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        if weight_decay > 0:
            # In Adam the decay would be rescaled like the gradient; AdamW
            # applies it to the weights directly.
            return torch.optim.AdamW(params, lr, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr, betas=(0.9, 0.999))
    raise ValueError(name)


def build_scheduler(optimizer, name, total_iters, warmup_iters=0):
    """Return the learning-rate scheduler of a run, stepped once per iteration.

    Args:
        optimizer (torch.optim.Optimizer): Optimiser whose learning rate is set.
        name (str): "constant", "step" or "poly".
        total_iters (int): Number of iterations of the whole run.
        warmup_iters (int): Number of iterations of the linear warm-up.

    Returns:
        torch.optim.lr_scheduler.LRScheduler or None: The scheduler; None for
            a constant rate without warm-up, which needs no scheduler.
    """
    T = total_iters - warmup_iters  # iterations of the schedule itself
    if name == "constant":
        main = None
    elif name == "step":  # the rate is multiplied by 0.1 at 0.5 T and at 0.75 T
        milestones = [int(0.5 * T), int(0.75 * T)]
        main = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif name == "poly":  # the rate is lr * (1 - t / T)^0.9
        main = PolynomialLR(optimizer, total_iters=T, power=0.9)
    else:
        raise ValueError(name)
    if warmup_iters == 0:
        return main
    # Warm-up (Task C): the rate first rises linearly from lr / warmup_iters to
    # lr, then the schedule starts. At the switch PyTorch prints a UserWarning
    # about scheduler.step(); it comes from SequentialLR and is harmless.
    warm = LinearLR(optimizer, start_factor=1 / warmup_iters, total_iters=warmup_iters)
    if main is None:
        return warm  # after the warm-up LinearLR keeps the rate at lr
    return SequentialLR(optimizer, [warm, main], milestones=[warmup_iters])


def fit(model, train_loader, val_loader, test_loader, args, device):
    """Train the model for args.epochs epochs and log every epoch.

    Writes results/<run>.csv (one row per epoch) and, after the last epoch,
    checkpoints/<run>.pt; <run> is for example "cifar10_adam_step_seed0".
    The logged "lr" is the learning rate at the start of the epoch; the
    scheduler changes it at every iteration.

    Args:
        model (nn.Module): Network to train, already on device.
        train_loader (DataLoader): Shuffled, augmented training batches.
        val_loader (DataLoader): Validation batches.
        test_loader (DataLoader): Test batches, logged for the learning
            curves only, never used to choose a model.
        args (argparse.Namespace): The arguments of parameters.py.
        device (str): "cpu" or "cuda".
    """
    optimizer = build_optimizer(
        model.parameters(), args.optimizer, args.lr, WEIGHT_DECAY[args.task]
    )
    # The scheduler counts iterations, not epochs.
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        total_iters=args.epochs * iters_per_epoch,
        warmup_iters=WARMUP_EPOCHS[args.task] * iters_per_epoch,
    )
    run = run_name(args.task, args.optimizer, args.scheduler, args.seed)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]  # the rate the epoch starts with
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, optimizer, device, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": lr,
                "grad_norm": grad_norm,
                "time": time.time() - start,
            }
        )
        print(", ".join(f"{k} {v:.4g}" for k, v in log[-1].items()))
        # The file is rewritten every epoch: an interrupted run keeps its log.
        save_log(f"results/{run}.csv", log)
    save_checkpoint(f"checkpoints/{run}.pt", model, optimizer, args.epochs)
