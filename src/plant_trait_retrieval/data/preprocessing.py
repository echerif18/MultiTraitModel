"""Preprocessing utilities.

- Differentiable static Box-Cox transformation layers for targets.
- Optional spectra standardization.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
class StaticTransformationLayer(nn.Module):
    """Apply a fixed transformation function as a module."""

    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformation(x)


class BoxCoxTransform(nn.Module):
    """Per-trait Box-Cox transform with optional normalization and inverse."""

    def __init__(
        self,
        lambda_values: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("lambda_values", lambda_values.float())
        if mean is None:
            mean = torch.zeros_like(self.lambda_values)
        if std is None:
            std = torch.ones_like(self.lambda_values)
        if shift is None:
            shift = torch.zeros_like(self.lambda_values)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.register_buffer("shift", shift.float())
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(x, torch.nan)
        for i in range(x.size(1)):
            col = x[:, i]
            mask = torch.isfinite(col)
            if mask.sum() == 0:
                continue
            val = col[mask] + self.shift[i]
            val = torch.clamp(val, min=self.eps)
            lam = self.lambda_values[i]
            if torch.isclose(lam, torch.tensor(0.0, device=lam.device)):
                trans = torch.log(val)
            else:
                trans = (torch.pow(val, lam) - 1.0) / lam
            trans = (trans - self.mean[i]) / (self.std[i] + self.eps)
            out[mask, i] = trans
        return out

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(y, torch.nan)
        for i in range(y.size(1)):
            col = y[:, i]
            mask = torch.isfinite(col)
            if mask.sum() == 0:
                continue
            val = col[mask] * (self.std[i] + self.eps) + self.mean[i]
            lam = self.lambda_values[i]
            if torch.isclose(lam, torch.tensor(0.0, device=lam.device)):
                orig = torch.exp(val)
            else:
                orig = torch.pow(lam * val + 1.0, 1.0 / lam)
            orig = orig - self.shift[i]
            out[mask, i] = orig
        return out


class PowerTransformerWrapper:
    """Differentiable static Box-Cox wrapper (keeps existing API name)."""

    def __init__(self, method: str = "box-cox") -> None:
        self.method = method
        self._fitted = False
        self.lambda_values: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.shift: Optional[np.ndarray] = None
        self.boxcox_layer: Optional[BoxCoxTransform] = None
        self.transformation_layer: Optional[StaticTransformationLayer] = None

    # ------------------------------------------------------------------
    def fit(self, targets: np.ndarray) -> "PowerTransformerWrapper":
        """Fit per-trait Box-Cox params on finite values."""
        arr = np.asarray(targets, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"targets must be 2D, got shape {arr.shape}")
        n_traits = int(arr.shape[1])
        lambdas = np.zeros(n_traits, dtype=np.float32)
        means = np.zeros(n_traits, dtype=np.float32)
        stds = np.ones(n_traits, dtype=np.float32)
        shifts = np.zeros(n_traits, dtype=np.float32)

        for t in range(n_traits):
            col = arr[:, t]
            mask = np.isfinite(col)
            if mask.sum() < 2:
                continue
            vals = col[mask].astype(np.float64)
            min_v = np.min(vals)
            shift = max(0.0, -min_v + 1e-6)
            vals_pos = vals + shift

            try:
                lam = float(stats.boxcox_normmax(vals_pos, method="mle"))
            except Exception:
                lam = 0.0
            if abs(lam) < 1e-12:
                transformed = np.log(vals_pos)
            else:
                transformed = (np.power(vals_pos, lam) - 1.0) / lam
            mu = float(np.mean(transformed))
            sd = float(np.std(transformed))
            if sd <= 0:
                sd = 1.0

            lambdas[t] = lam
            means[t] = mu
            stds[t] = sd
            shifts[t] = shift

        self.lambda_values = lambdas
        self.mean = means
        self.std = stds
        self.shift = shifts
        self.boxcox_layer = BoxCoxTransform(
            lambda_values=torch.from_numpy(self.lambda_values),
            mean=torch.from_numpy(self.mean),
            std=torch.from_numpy(self.std),
            shift=torch.from_numpy(self.shift),
        )
        self.transformation_layer = StaticTransformationLayer(self.boxcox_layer)
        self._fitted = True
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        assert self._fitted, "Must fit before transform"
        arr = np.asarray(targets, dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(arr)
            out = self.transformation_layer(t).cpu().numpy().astype(np.float32)
        return out

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        assert self._fitted, "Must fit before inverse_transform"
        arr = np.asarray(targets, dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(arr)
            out = self.boxcox_layer.inverse(t).cpu().numpy().astype(np.float32)
        return out

    def inverse_transform_tensor(self, t: torch.Tensor) -> np.ndarray:
        assert self._fitted, "Must fit before inverse_transform_tensor"
        with torch.no_grad():
            return self.boxcox_layer.inverse(t).cpu().numpy().astype(np.float32)

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "Must fit before transform_tensor"
        return self.transformation_layer(t)

    # Convenience: transform as callable for Dataset ---------------------
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return self.transform_tensor(y)

    # Persistence --------------------------------------------------------
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "PowerTransformerWrapper":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        return obj


# ---------------------------------------------------------------------------
class SpectraStandardScaler:
    """Per-band z-score normalisation fitted on training data."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, spectra: np.ndarray) -> "SpectraStandardScaler":
        self.scaler.fit(spectra)
        self._fitted = True
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        return self.scaler.transform(spectra).astype(np.float32)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "SpectraStandardScaler":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


# ---------------------------------------------------------------------------
def build_transforms(
    train_spectra: np.ndarray,
    train_targets: np.ndarray,
    power_method: str = "yeo-johnson",
    scale_spectra: bool = True,
    save_dir: Optional[Path] = None,
) -> tuple[Optional[SpectraStandardScaler], PowerTransformerWrapper]:
    """Fit scalers on training data and optionally persist them.

    Returns
    -------
    spectra_scaler : SpectraStandardScaler
    target_scaler  : PowerTransformerWrapper
    """
    spectra_scaler = SpectraStandardScaler().fit(train_spectra) if scale_spectra else None
    target_scaler = PowerTransformerWrapper(method=power_method).fit(train_targets)

    if save_dir is not None:
        save_dir = Path(save_dir)
        if spectra_scaler is not None:
            spectra_scaler.save(save_dir / "spectra_scaler.pkl")
        target_scaler.save(save_dir / "target_scaler.pkl")

    return spectra_scaler, target_scaler
