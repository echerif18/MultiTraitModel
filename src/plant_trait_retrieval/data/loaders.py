"""DataLoader factory."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import HyperspectralDataset
from .preprocessing import SpectraStandardScaler, PowerTransformerWrapper


def make_loaders(
    train_spectra: np.ndarray,
    train_targets: np.ndarray,
    val_spectra: np.ndarray,
    val_targets: np.ndarray,
    spectra_scaler: Optional[SpectraStandardScaler],
    target_scaler: PowerTransformerWrapper,
    transform_targets_in_loader: bool = False,
    train_sample_weights: Optional[np.ndarray] = None,
    train_augmentation: bool = False,
    aug_prob: float = 0.5,
    betashift: float = 0.01,
    slopeshift: float = 0.01,
    multishift: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with scaling applied."""
    # Scale spectra optionally
    if spectra_scaler is None:
        train_X = train_spectra.astype(np.float32)
        val_X = val_spectra.astype(np.float32)
    else:
        train_X = spectra_scaler.transform(train_spectra)
        val_X = spectra_scaler.transform(val_spectra)
    if transform_targets_in_loader:
        train_Y = target_scaler.transform(train_targets)
        val_Y = target_scaler.transform(val_targets)
    else:
        train_Y = train_targets.astype(np.float32)
        val_Y = val_targets.astype(np.float32)

    train_ds = HyperspectralDataset(
        train_X,
        train_Y,
        sample_weights=train_sample_weights,
        augmentation=train_augmentation,
        aug_prob=aug_prob,
        betashift=betashift,
        slopeshift=slopeshift,
        multishift=multishift,
    )
    val_ds = HyperspectralDataset(val_X, val_Y)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
