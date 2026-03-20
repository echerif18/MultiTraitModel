"""SHAP-based feature importance utilities for 1D spectral models."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _TraitHead(nn.Module):
    """Wrap a multi-output model to expose one trait as scalar output."""

    def __init__(self, model: nn.Module, trait_idx: int) -> None:
        super().__init__()
        self.model = model
        self.trait_idx = int(trait_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y[:, self.trait_idx : self.trait_idx + 1]


def _to_band_importance(shap_values: object) -> np.ndarray:
    """Convert SHAP output to per-band mean absolute importance."""
    arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        # Common shape for 1D CNN inputs: (N, C, L)
        if arr.shape[1] == 1:
            arr = arr[:, 0, :]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = np.mean(arr, axis=1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")
    return np.mean(np.abs(arr), axis=0)


@torch.no_grad()
def _predict_trait(model: nn.Module, x: np.ndarray, trait_idx: int, device: str) -> np.ndarray:
    model = model.to(device).eval()
    xb = torch.from_numpy(x.astype(np.float32)).to(device)
    y = model(xb)[:, int(trait_idx)].detach().cpu().numpy()
    return y


def compute_shap_importance(
    model: nn.Module,
    background_spectra: np.ndarray,
    eval_spectra: np.ndarray,
    wavelengths: Sequence[float],
    trait_names: Sequence[str],
    trait_indices: Sequence[int],
    output_dir: Path,
    device: str = "cpu",
    n_samples: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run SHAP attribution for selected traits and export CSV/plots.

    Parameters
    ----------
    model             : trained torch model, output shape (N, n_traits)
    background_spectra: (N_bg, L) transformed spectra
    eval_spectra      : (N_eval, L) transformed spectra
    wavelengths       : list-like length L
    trait_names       : full trait names list
    trait_indices     : which traits to explain
    output_dir        : folder to save artifacts
    device            : cuda/cpu
    n_samples         : SHAP nsamples parameter
    """
    try:
        import shap
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "SHAP is not installed. Install with: poetry add shap"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wl = np.asarray(wavelengths, dtype=np.float32)
    if wl.shape[0] != background_spectra.shape[1]:
        raise ValueError("wavelengths length must match spectra band count.")

    bg = torch.from_numpy(background_spectra.astype(np.float32)).unsqueeze(1).to(device)
    ev = torch.from_numpy(eval_spectra.astype(np.float32)).unsqueeze(1).to(device)

    rows = []
    per_trait = []

    for trait_idx in trait_indices:
        tname = str(trait_names[int(trait_idx)])
        wrapped = _TraitHead(model, int(trait_idx)).to(device).eval()
        explainer = shap.GradientExplainer(wrapped, bg)
        shap_vals = explainer.shap_values(ev, nsamples=int(n_samples))
        band_imp = _to_band_importance(shap_vals)

        trait_df = pd.DataFrame(
            {
                "trait_idx": int(trait_idx),
                "trait": tname,
                "wavelength": wl,
                "mean_abs_shap": band_imp,
            }
        )
        per_trait.append(trait_df)
        rows.append(band_imp)

        plt.figure(figsize=(11, 4))
        plt.plot(wl, band_imp, lw=1.4)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mean |SHAP|")
        plt.title(f"SHAP spectral importance - {tname}")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_importance_{tname}.png", dpi=170)
        plt.close()

    by_trait = pd.concat(per_trait, ignore_index=True)
    by_trait.to_csv(output_dir / "shap_mean_abs_by_trait.csv", index=False)

    global_imp = np.mean(np.stack(rows, axis=0), axis=0)
    global_df = pd.DataFrame(
        {
            "wavelength": wl,
            "mean_abs_shap_global": global_imp,
        }
    )
    global_df.to_csv(output_dir / "shap_mean_abs_global.csv", index=False)

    plt.figure(figsize=(11, 4))
    plt.plot(wl, global_imp, lw=1.6)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean |SHAP|")
    plt.title("Global SHAP spectral importance")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_global.png", dpi=170)
    plt.close()

    return by_trait, global_df
