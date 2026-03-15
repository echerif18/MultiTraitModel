"""Optimiser and LR-scheduler factories."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR,
    SequentialLR,
)


def build_optimizer(model: nn.Module, cfg: DictConfig) -> Optimizer:
    return AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


def build_scheduler(optimizer: Optimizer, cfg: DictConfig):
    """Return (scheduler, use_metric) where use_metric=True means ReduceLROnPlateau."""
    name = cfg.training.scheduler.lower()
    epochs = cfg.training.epochs
    warmup = cfg.training.warmup_epochs

    if name == "cosine":
        if warmup > 0:
            def warmup_lambda(epoch):
                if epoch < warmup:
                    return float(epoch + 1) / float(warmup + 1)
                return 1.0

            warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=epochs - warmup, eta_min=1e-7
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        return scheduler, False

    elif name == "step":
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        return scheduler, False

    elif name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return scheduler, True

    else:
        raise ValueError(f"Unknown scheduler '{name}'")
