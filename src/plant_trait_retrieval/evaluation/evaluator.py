"""Evaluate a trained model on a held-out test set."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.preprocessing import PowerTransformerWrapper, SpectraStandardScaler
from ..data.dataset import HyperspectralDataset
from .metrics import compute_metrics, eval_metrics_table


@torch.no_grad()
def predict(
    model: nn.Module,
    spectra: np.ndarray,
    spectra_scaler: Optional[SpectraStandardScaler],
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run inference on raw spectra, return predictions in transformed space.

    Parameters
    ----------
    model         : trained EfficientNet1d
    spectra       : (N, L) raw spectra
    spectra_scaler: fitted SpectraStandardScaler (or None for raw spectra)
    batch_size    : inference batch size
    device        : 'cuda' | 'cpu'

    Returns
    -------
    preds : (N, T) float32 in transformed target space
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    X = spectra_scaler.transform(spectra) if spectra_scaler is not None else spectra.astype(np.float32)
    dummy_y = np.zeros((len(X), 1), dtype=np.float32)
    ds = HyperspectralDataset(X, dummy_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    for x, _ in loader:
        x = x.to(dev)
        out = model(x)
        all_preds.append(out.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def evaluate(
    model: nn.Module,
    spectra: np.ndarray,
    targets: np.ndarray,
    spectra_scaler: Optional[SpectraStandardScaler],
    target_scaler: PowerTransformerWrapper,
    trait_names: Optional[List[str]] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    prefix: str = "test",
) -> pd.DataFrame:
    """Predict and compute metrics in the original (inverse-transformed) scale.

    Returns
    -------
    metrics_df : pd.DataFrame with per-trait and mean metrics
    """
    preds_transformed = predict(model, spectra, spectra_scaler, batch_size, device)
    preds_original = target_scaler.inverse_transform(preds_transformed)
    targets_original = target_scaler.inverse_transform(
        target_scaler.transform(targets)  # round-trip to apply same transform
    )

    metrics_df = compute_metrics(targets_original, preds_original, trait_names)
    metrics_table_df = eval_metrics_table(targets_original, preds_original, trait_names)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out / f"{prefix}_metrics.csv", index=False)
        metrics_table_df.to_csv(out / f"{prefix}_metrics_table.csv", index=True)
        csv_cols = trait_names or [f"trait_{i}" for i in range(preds_original.shape[1])]
        pred_df = pd.DataFrame(preds_original, columns=csv_cols)
        obs_df = pd.DataFrame(targets_original, columns=csv_cols)
        pred_df.to_csv(out / f"{prefix}_predictions.csv", index=False)
        obs_df.to_csv(out / f"{prefix}_observations.csv", index=False)
        np.save(out / f"{prefix}_predictions.npy", preds_original)
        np.save(out / f"{prefix}_targets.npy", targets_original)

    return metrics_df
