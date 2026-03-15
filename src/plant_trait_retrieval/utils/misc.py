"""Utility helpers."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = __name__) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(cfg: DictConfig, run_name: Optional[str] = None, fold: Optional[int] = None) -> None:
    """Initialise a W&B run."""
    if not cfg.logging.wandb.enabled:
        return
    name = run_name or cfg.project.name
    if fold is not None:
        name = f"{name}_fold{fold}"
    wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity or None,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.logging.wandb.tags),
        reinit=True,
    )


def finish_wandb() -> None:
    if wandb.run is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_system_info(logger: logging.Logger) -> None:
    logger.info(f"PyTorch version : {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device     : {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version    : {torch.version.cuda}")
    else:
        logger.info("Running on CPU")
