"""scripts/aggregate_results.py

Aggregate per-fold metric CSVs into a summary table and produce
per-trait R² / RMSE bar charts.

Usage
-----
    python scripts/aggregate_results.py \
        --results_dir results/cv \
        --output_dir results/figures \
        --n_folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_fold_metrics(results_dir: Path, n_folds: int) -> pd.DataFrame:
    dfs = []
    for fold in range(n_folds):
        pattern = list((results_dir / f"fold_{fold}" / "eval").glob("*_metrics.csv"))
        if not pattern:
            print(f"  [warn] No metrics CSV found for fold {fold}, skipping.")
            continue
        df = pd.read_csv(pattern[0])
        df["fold"] = fold
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No metric files found under {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def summary_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["rmse", "mae", "r2", "nrmse", "bias"]
    grp = (
        all_metrics[all_metrics["trait"] != "MEAN"]
        .groupby("trait")[numeric_cols]
        .agg(["mean", "std"])
    )
    # Flatten multi-level columns
    grp.columns = [f"{c}_{s}" for c, s in grp.columns]
    return grp.reset_index()


def plot_r2_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    traits = summary["trait"].tolist()
    r2_mean = summary["r2_mean"].values
    r2_std = summary["r2_std"].values

    x = np.arange(len(traits))
    bars = ax.bar(x, r2_mean, yerr=r2_std, capsize=4,
                  color=sns.color_palette("muted", len(traits)),
                  edgecolor="black", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(traits, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R²  (mean ± std across folds)")
    ax.set_title("Per-trait R² – 5-Fold Cross-Validation")
    ax.set_ylim(bottom=min(-0.1, r2_mean.min() - r2_std.max() - 0.05))
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "r2_per_trait.png", dpi=150)
    plt.close(fig)
    print(f"Saved R² bar chart → {output_dir / 'r2_per_trait.png'}")


def plot_rmse_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    traits = summary["trait"].tolist()
    rmse_mean = summary["rmse_mean"].values
    rmse_std = summary["rmse_std"].values

    x = np.arange(len(traits))
    ax.bar(x, rmse_mean, yerr=rmse_std, capsize=4,
           color=sns.color_palette("rocket_r", len(traits)),
           edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(traits, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE  (mean ± std across folds)")
    ax.set_title("Per-trait RMSE – 5-Fold Cross-Validation")
    plt.tight_layout()
    fig.savefig(output_dir / "rmse_per_trait.png", dpi=150)
    plt.close(fig)
    print(f"Saved RMSE bar chart → {output_dir / 'rmse_per_trait.png'}")


def plot_r2_heatmap(all_metrics: pd.DataFrame, output_dir: Path) -> None:
    """R² heatmap: traits × folds."""
    pivot = (
        all_metrics[all_metrics["trait"] != "MEAN"]
        .pivot(index="trait", columns="fold", values="r2")
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.45)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.3, ax=ax,
        cbar_kws={"label": "R²"},
    )
    ax.set_title("R² per trait per fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_dir / "r2_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved R² heatmap → {output_dir / 'r2_heatmap.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/cv")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("Loading fold metrics …")
    all_metrics = load_fold_metrics(results_dir, args.n_folds)

    summary = summary_table(all_metrics)
    summary_path = results_dir / "cv_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table → {summary_path}")
    print(summary.to_string(index=False))

    plot_r2_bar(summary, output_dir)
    plot_rmse_bar(summary, output_dir)
    plot_r2_heatmap(all_metrics, output_dir)


if __name__ == "__main__":
    main()
