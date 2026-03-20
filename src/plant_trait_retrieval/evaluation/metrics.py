"""Evaluation metrics for multi-trait regression.

Metrics (per-trait and global)
-------------------------------
- RMSE   Root Mean Squared Error
- MAE    Mean Absolute Error
- R²     Coefficient of determination
- NRMSE  Normalised RMSE (by trait range or std)
- Bias   Mean signed error
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per-trait regression metrics.

    Parameters
    ----------
    y_true       : (N, T) ground-truth values (original scale, inverse-transformed)
    y_pred       : (N, T) predictions (original scale)
    trait_names  : list of T trait names (optional)

    Returns
    -------
    pd.DataFrame with columns [trait, rmse, mae, r2, nrmse, bias]
    """
    n, T = y_true.shape
    if trait_names is None:
        trait_names = [f"trait_{i}" for i in range(T)]

    records = []
    for t in range(T):
        yt = y_true[:, t].astype(float)
        yp = y_pred[:, t].astype(float)
        finite_mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite_mask]
        yp = yp[finite_mask]

        if yt.size == 0:
            records.append(
                dict(
                    trait=trait_names[t],
                    rmse=np.nan,
                    mae=np.nan,
                    r2=np.nan,
                    nrmse=np.nan,
                    bias=np.nan,
                )
            )
            continue

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        # r2_score needs at least 2 samples
        r2 = float(r2_score(yt, yp)) if yt.size > 1 else np.nan
        q95 = float(np.nanquantile(yt, 0.95))
        q05 = float(np.nanquantile(yt, 0.05))
        denom = q95 - q05
        nrmse = rmse / denom if denom > 0 else np.nan
        bias = float(np.mean(yp - yt))

        records.append(
            dict(trait=trait_names[t], rmse=rmse, mae=mae, r2=r2, nrmse=nrmse, bias=bias)
        )

    df = pd.DataFrame(records)
    # Append global average row
    avg = df.drop(columns="trait").mean().to_dict()
    avg["trait"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df


def metrics_to_wandb(metrics_df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """Flatten metrics DataFrame to a flat dict suitable for wandb.log()."""
    log = {}
    for _, row in metrics_df.iterrows():
        name = row["trait"]
        for col in ["rmse", "mae", "r2", "nrmse", "bias"]:
            key = f"{prefix}/{name}/{col}" if prefix else f"{name}/{col}"
            log[key] = row[col]
    return log


def eval_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build per-trait metrics table similar to the original eval_metrics output.

    Returns a DataFrame indexed by trait with columns:
    ['r2_score', 'RMSE', 'nRMSE (%)', 'MAE', 'Bias', 'RPD', 'spearmanr_squared'].
    """
    _, n_traits = y_true.shape
    if trait_names is None:
        trait_names = [f"trait_{i}" for i in range(n_traits)]

    rows = []
    for t, trait in enumerate(trait_names):
        yt = pd.Series(y_true[:, t]).reset_index(drop=True)
        yp = pd.Series(y_pred[:, t]).reset_index(drop=True)

        invalid_idx = np.union1d(np.where(~np.isfinite(yt.values))[0], np.where(~np.isfinite(yp.values))[0])
        if len(invalid_idx) > 0:
            yt = yt.drop(invalid_idx).reset_index(drop=True)
            yp = yp.drop(invalid_idx).reset_index(drop=True)

        if yt.notnull().sum() == 0:
            rows.append(
                {
                    "trait": trait,
                    "r2_score": np.nan,
                    "RMSE": np.nan,
                    "nRMSE (%)": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "RPD": np.nan,
                    "spearmanr_squared": np.nan,
                }
            )
            continue

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        q95 = np.nanquantile(np.array(yt), 0.95)
        q05 = np.nanquantile(np.array(yt), 0.05)
        denom = q95 - q05
        nrmse_percent = float((rmse * 100.0 / denom) if denom != 0 else np.nan)
        # Avoid scipy warning on constant arrays; correlation is undefined there.
        if np.nanstd(np.array(yt)) == 0 or np.nanstd(np.array(yp)) == 0:
            spearman_sq = np.nan
        else:
            spearman = stats.spearmanr(yt, yp).statistic
            spearman_sq = float(spearman**2) if spearman is not None else np.nan

        rows.append(
            {
                "trait": trait,
                "r2_score": float(r2_score(yt, yp)),
                "RMSE": rmse,
                "nRMSE (%)": nrmse_percent,
                "MAE": float(mean_absolute_error(yt, yp)),
                "Bias": float(np.sum(np.array(yt) - np.array(yp)) / len(yp)),
                "RPD": float(np.std(np.array(yt), ddof=1) / rmse) if rmse != 0 else np.nan,
                "spearmanr_squared": spearman_sq,
            }
        )

    return pd.DataFrame(rows).set_index("trait")
