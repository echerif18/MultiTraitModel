"""HyperspectralDataset – PyTorch Dataset for tabular hyperspectral + trait data."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    """Dataset for 1-D hyperspectral spectra and multi-trait regression targets.

    Parameters
    ----------
    spectra : np.ndarray  shape (N, n_bands)
    targets : np.ndarray  shape (N, n_traits)
    spectra_transform : callable, optional
        Applied to each spectrum (after conversion to float32 tensor).
    target_transform : callable, optional
        Applied to each target vector.
    """

    def __init__(
        self,
        spectra: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        augmentation: bool = False,
        aug_prob: float = 0.5,
        betashift: float = 0.01,
        slopeshift: float = 0.01,
        multishift: float = 0.1,
        spectra_transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        assert len(spectra) == len(targets), "spectra and targets must have same length"
        self.spectra = torch.from_numpy(spectra.astype(np.float32))      # (N, L)
        self.targets = torch.from_numpy(targets.astype(np.float32))      # (N, T)
        self.sample_weights = (
            torch.from_numpy(sample_weights.astype(np.float32))
            if sample_weights is not None
            else None
        )
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.betashift = betashift
        self.slopeshift = slopeshift
        self.multishift = multishift
        self.spectra_transform = spectra_transform
        self.target_transform = target_transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int):
        x = self.spectra[idx]  # (L,)
        y = self.targets[idx]

        if self.augmentation and random.random() < self.aug_prob:
            x = self._apply_augmentation(x)

        if self.spectra_transform is not None:
            x = self.spectra_transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        x = x.unsqueeze(0)  # (1, L) – channel-first for Conv1d

        if self.sample_weights is not None:
            return x, y, self.sample_weights[idx]
        return x, y

    def _apply_augmentation(self, x_tensor: torch.Tensor) -> torch.Tensor:
        augmentation_methods = [self._add_noise, self._shift]
        aug_method = random.choice(augmentation_methods)
        return aug_method(x_tensor)

    def _add_noise(self, x_tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        noise = torch.randn_like(x_tensor) * std
        return x_tensor + noise

    def _shift(self, x_tensor: torch.Tensor) -> torch.Tensor:
        std = torch.std(x_tensor)
        beta = (torch.rand(1) * 2 * self.betashift - self.betashift) * std
        slope = (torch.rand(1) * 2 * self.slopeshift - self.slopeshift + 1)
        axis = torch.arange(x_tensor.shape[0], dtype=torch.float32) / float(x_tensor.shape[0])
        offset = slope * axis + beta - axis - slope / 2.0 + 0.5
        multi = torch.rand(1) * 2 * self.multishift - self.multishift + 1
        augmented_x = multi * x_tensor + offset * std
        return torch.clamp(augmented_x, min=0.0)

    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        spectra_cols: list[str],
        trait_cols: list[str],
        sample_weight_col: Optional[str] = None,
        augmentation: bool = False,
        aug_prob: float = 0.5,
        betashift: float = 0.01,
        slopeshift: float = 0.01,
        multishift: float = 0.1,
        spectra_transform=None,
        target_transform=None,
    ) -> "HyperspectralDataset":
        spectra = df[spectra_cols].values
        targets = df[trait_cols].values
        sample_weights = None
        if sample_weight_col is not None and sample_weight_col in df.columns:
            sample_weights = df[sample_weight_col].values
        return cls(
            spectra,
            targets,
            sample_weights,
            augmentation,
            aug_prob,
            betashift,
            slopeshift,
            multishift,
            spectra_transform,
            target_transform,
        )

    @classmethod
    def from_indices(
        cls,
        spectra: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        augmentation: bool = False,
        aug_prob: float = 0.5,
        betashift: float = 0.01,
        slopeshift: float = 0.01,
        multishift: float = 0.1,
        spectra_transform=None,
        target_transform=None,
    ) -> "HyperspectralDataset":
        return cls(
            spectra[indices],
            targets[indices],
            sample_weights[indices] if sample_weights is not None else None,
            augmentation,
            aug_prob,
            betashift,
            slopeshift,
            multishift,
            spectra_transform,
            target_transform,
        )
