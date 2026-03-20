"""Aggregate transferability outputs across all held-out domains.

Creates:
1) Ordered merged CSV of obs/pred from all domains.
2) Global scatter plots (obs vs pred) for all traits.
3) Per-trait + macro metrics table (R² and nRMSE%).

Usage
-----
python scripts/aggregate_transfer_results.py \
  --transfer_dir results/transfer/transfer_all_20260315_224236
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def _natural_domain_key(x: str):
    return (0, int(x)) if x.isdigit() else (1, x)


def find_latest_transfer_dir(root: Path) -> Path:
    candidates = sorted(root.glob("transfer_all_*"))
    if not candidates:
        raise FileNotFoundError(f"No transfer_all_* directories found in {root}")
    return candidates[-1]


def _domain_in_range(name: str, domain_min: int | None, domain_max: int | None) -> bool:
    if domain_min is None and domain_max is None:
        return True
    if not name.isdigit():
        return False
    v = int(name)
    if domain_min is not None and v < domain_min:
        return False
    if domain_max is not None and v > domain_max:
        return False
    return True


def load_domain_frames(
    transfer_dir: Path,
    domain_min: int | None = None,
    domain_max: int | None = None,
) -> tuple[list[pd.DataFrame], list[str]]:
    domain_frames = []
    trait_names = None
    domain_dirs = sorted(
        [
            p
            for p in transfer_dir.iterdir()
            if p.is_dir() and _domain_in_range(p.name, domain_min=domain_min, domain_max=domain_max)
        ],
        key=lambda p: _natural_domain_key(p.name),
    )

    for domain_dir in domain_dirs:
        pred_path = domain_dir / "eval" / "target_domain_predictions.csv"
        obs_path = domain_dir / "eval" / "target_domain_observations.csv"
        if not pred_path.exists() or not obs_path.exists():
            continue

        pred = pd.read_csv(pred_path)
        obs = pd.read_csv(obs_path)
        if trait_names is None:
            trait_names = list(obs.columns)

        n = min(len(pred), len(obs))
        pred = pred.iloc[:n].reset_index(drop=True)
        obs = obs.iloc[:n].reset_index(drop=True)

        merged = pd.DataFrame(
            {
                "target_domain": domain_dir.name,
                "row_in_domain": np.arange(n, dtype=int),
            }
        )
        for c in trait_names:
            merged[f"obs__{c}"] = obs[c]
            merged[f"pred__{c}"] = pred[c]
        domain_frames.append(merged)

    if not domain_frames:
        raise FileNotFoundError(
            f"No domain eval CSV pairs found under {transfer_dir} "
            f"for domain range [{domain_min}, {domain_max}]"
        )
    return domain_frames, trait_names or []


def per_trait_metrics(merged_df: pd.DataFrame, trait_names: list[str]) -> pd.DataFrame:
    rows = []
    for trait in trait_names:
        y_true = merged_df[f"obs__{trait}"].to_numpy(dtype=float)
        y_pred = merged_df[f"pred__{trait}"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]
        if yt.size < 2:
            rows.append({"trait": trait, "r2": np.nan, "rmse": np.nan, "nrmse_percent": np.nan, "n": int(yt.size)})
            continue

        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        q01, q99 = np.nanquantile(yt, 0.01), np.nanquantile(yt, 0.99)
        denom = q99 - q01
        nrmse = float((rmse * 100.0 / denom) if denom != 0 else np.nan)
        rows.append({"trait": trait, "r2": r2, "rmse": rmse, "nrmse_percent": nrmse, "n": int(yt.size)})

    out = pd.DataFrame(rows)
    macro = {
        "trait": "MACRO_MEAN",
        "r2": float(np.nanmean(out["r2"])),
        "rmse": float(np.nanmean(out["rmse"])),
        "nrmse_percent": float(np.nanmean(out["nrmse_percent"])),
        "n": int(np.nansum(out["n"])),
    }
    out = pd.concat([out, pd.DataFrame([macro])], ignore_index=True)
    return out


def plot_global_scatter(
    merged_df: pd.DataFrame,
    trait_names: list[str],
    metrics_df: pd.DataFrame,
    out_path: Path,
) -> None:
    n_traits = len(trait_names)
    n_cols = 5
    n_rows = int(np.ceil(n_traits / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, trait in enumerate(trait_names):
        ax = axes[i]
        y_true = merged_df[f"obs__{trait}"].to_numpy(dtype=float)
        y_pred = merged_df[f"pred__{trait}"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]

        # x = predictions, y = observations
        ax.scatter(yp, yt, s=8, alpha=0.25)
        if yt.size > 0:
            lo = np.nanmin([yt.min(), yp.min()])
            hi = np.nanmax([yt.max(), yp.max()])
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        mrow = metrics_df[metrics_df["trait"] == trait]
        if not mrow.empty:
            r2 = mrow.iloc[0]["r2"]
            nrmse = mrow.iloc[0]["nrmse_percent"]
            ax.text(
                0.03,
                0.97,
                f"R²={r2:.3f}\nnRMSE={nrmse:.2f}%",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
            )
        ax.set_title(trait, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")

    for j in range(n_traits, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Global Transferability Scatter: Observed vs Predicted (All Domains)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer_root", default="results/transfer", help="Root containing transfer_all_* runs.")
    parser.add_argument("--transfer_dir", default=None, help="Specific transfer run directory.")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: <transfer_dir>/global_aggregate).")
    parser.add_argument("--domain_min", type=int, default=None, help="Minimum numeric target_domain to include.")
    parser.add_argument("--domain_max", type=int, default=None, help="Maximum numeric target_domain to include.")
    args = parser.parse_args()

    root = Path(args.transfer_root)
    transfer_dir = Path(args.transfer_dir) if args.transfer_dir else find_latest_transfer_dir(root)
    suffix = ""
    if args.domain_min is not None or args.domain_max is not None:
        suffix = f"_domains_{args.domain_min if args.domain_min is not None else 'min'}_{args.domain_max if args.domain_max is not None else 'max'}"
    output_dir = Path(args.output_dir) if args.output_dir else (transfer_dir / f"global_aggregate{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_frames, trait_names = load_domain_frames(
        transfer_dir,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
    )
    merged = pd.concat(domain_frames, ignore_index=True)
    merged = merged.sort_values(["target_domain", "row_in_domain"], key=lambda s: s.map(lambda x: _natural_domain_key(str(x)) if s.name == "target_domain" else x))
    merged.to_csv(output_dir / "merged_predictions_observations.csv", index=False)

    metrics = per_trait_metrics(merged, trait_names)
    metrics.to_csv(output_dir / "global_trait_metrics.csv", index=False)

    plot_global_scatter(merged, trait_names, metrics, output_dir / "global_obs_pred_scatter.png")

    print(f"Transfer dir used: {transfer_dir}")
    print(f"Merged CSV: {output_dir / 'merged_predictions_observations.csv'}")
    print(f"Metrics CSV: {output_dir / 'global_trait_metrics.csv'}")
    print(f"Scatter plot: {output_dir / 'global_obs_pred_scatter.png'}")
    macro = metrics[metrics['trait'] == 'MACRO_MEAN'].iloc[0]
    print(f"MACRO R2={macro['r2']:.4f} | MACRO nRMSE%={macro['nrmse_percent']:.4f}")


if __name__ == "__main__":
    main()
