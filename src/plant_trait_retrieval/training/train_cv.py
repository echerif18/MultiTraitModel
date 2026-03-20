"""5-fold cross-validation training script.

Usage
-----
    python -m plant_trait_retrieval.training.train_cv   # uses configs/base.yaml
    python -m plant_trait_retrieval.training.train_cv data.data_path=data/processed/dataset.csv
    python -m plant_trait_retrieval.training.train_cv training.epochs=300 training.batch_size=128
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from ..data.loaders import make_loaders
from ..data.preprocessing import build_transforms
from ..data.splitter import CVSplitter
from ..evaluation.evaluator import evaluate
from ..evaluation.metrics import metrics_to_wandb
from ..models.registry import build_model
from ..training.losses import build_loss
from ..training.trainer import Trainer
from ..utils.io import load_dataset
from ..utils.misc import (
    finish_wandb,
    get_device,
    get_logger,
    init_wandb,
    print_system_info,
    seed_everything,
)

console = Console()


def _compute_fold_sample_weights(
    df: pd.DataFrame,
    row_indices: np.ndarray,
    dataset_col: str = "dataset",
) -> np.ndarray:
    """Compute sample weights using inverse dataset frequency logic."""
    fold_df = df.iloc[row_indices].copy()
    dataset_counts = fold_df.groupby(dataset_col).size()
    wstr = 100.0 - 100.0 * (dataset_counts / float(len(fold_df)))
    weights = fold_df[dataset_col].map(wstr.to_dict()).to_numpy(dtype=np.float32)
    # Normalize around 1.0 to keep loss scale stable across folds.
    mean_w = float(np.mean(weights))
    if mean_w > 0:
        weights = weights / mean_w
    return weights


@hydra.main(config_path="../../../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(cfg.project.seed)
    device = get_device()
    print_system_info(logger)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Load data ────────────────────────────────────────────────────────
    spectra, targets, spectra_cols, trait_cols, df = load_dataset(cfg)
    original_row_idx = np.arange(len(df), dtype=np.int64)
    # Drop rows that cannot be used safely for training.
    valid_target_mask = np.isfinite(targets).any(axis=1)
    valid_spectra_mask = np.isfinite(spectra).all(axis=1)
    valid_row_mask = valid_target_mask & valid_spectra_mask
    dropped_bad_targets = int((~valid_target_mask).sum())
    dropped_bad_spectra = int((~valid_spectra_mask).sum())
    dropped_rows = int((~valid_row_mask).sum())

    filtered_df_export = df.loc[valid_row_mask].copy()
    filtered_df_export.insert(0, "original_row_index", original_row_idx[valid_row_mask])

    dropped_df_export = df.loc[~valid_row_mask].copy()
    if dropped_rows > 0:
        dropped_df_export.insert(0, "original_row_index", original_row_idx[~valid_row_mask])
        dropped_df_export.insert(1, "drop_all_nan_or_nonfinite_targets", (~valid_target_mask)[~valid_row_mask])
        dropped_df_export.insert(2, "drop_nonfinite_spectra", (~valid_spectra_mask)[~valid_row_mask])

    export_filtered_csv = cfg.data.get("export_filtered_csv", None)
    export_dropped_csv = cfg.data.get("export_dropped_csv", None)
    if export_filtered_csv:
        export_filtered_path = Path(str(export_filtered_csv))
        export_filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df_export.to_csv(export_filtered_path, index=False)
        logger.info(f"Exported filtered pre-CV dataset to: {export_filtered_path}")
    if export_dropped_csv and dropped_rows > 0:
        export_dropped_path = Path(str(export_dropped_csv))
        export_dropped_path.parent.mkdir(parents=True, exist_ok=True)
        dropped_df_export.to_csv(export_dropped_path, index=False)
        logger.info(f"Exported dropped-row report to: {export_dropped_path}")

    if dropped_rows > 0:
        spectra = spectra[valid_row_mask]
        targets = targets[valid_row_mask]
        df = df.loc[valid_row_mask].reset_index(drop=True)
        logger.warning(
            "Dropped unusable rows before CV split: "
            f"all-NaN/non-finite targets={dropped_bad_targets}, "
            f"non-finite spectra={dropped_bad_spectra}, total_dropped={dropped_rows}."
        )
    logger.info(f"Dataset: {spectra.shape[0]} samples | "
                f"{spectra.shape[1]} bands | {targets.shape[1]} traits")

    # ── CV splits ────────────────────────────────────────────────────────
    splits_dir = Path(cfg.data.cv_splits_dir) if cfg.data.cv_splits_dir else None
    if dropped_rows > 0 and splits_dir is not None:
        logger.warning(
            "Using filtered data after dropping rows, so precomputed split files are ignored for this run. "
            "Regenerate splits if you want persisted filtered folds."
        )
        splits_dir = None
    splitter = CVSplitter(
        n_folds=cfg.data.n_folds,
        splits_dir=splits_dir,
        strategy=cfg.data.cv_split_strategy,
        seed=cfg.project.seed,
    )

    fold_metrics = []
    exp_id = cfg.get("experiment_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("results/cv") / f"cv_{exp_id}"
    output_root.mkdir(parents=True, exist_ok=True)
    data_tag = Path(cfg.data.data_path).stem
    ckpt_root = Path(cfg.logging.checkpoint_dir) / "cv" / cfg.project.name / data_tag / f"models_{exp_id}"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"CV experiment id: {exp_id}")
    logger.info(f"CV results root: {output_root}")
    logger.info(f"CV checkpoint root: {ckpt_root}")

    stratify_labels = None
    if cfg.data.cv_split_strategy == "stratified":
        stratify_labels = df[cfg.data.cv_stratify_col].astype(str).values

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(len(spectra), stratify_labels=stratify_labels)
    ):
        logger.info(f"\n{'='*60}\n  Fold {fold+1}/{cfg.data.n_folds}  "
                    f"(train={len(train_idx)}, val={len(val_idx)})\n{'='*60}")

        # Init W&B run per fold
        init_wandb(cfg, fold=fold)

        # ── Preprocessing (fit only on train) ────────────────────────────
        fold_artifact_dir = output_root / f"fold_{fold}"
        spectra_scaler, target_scaler = build_transforms(
            train_spectra=spectra[train_idx],
            train_targets=targets[train_idx],
            power_method=cfg.data.power_method,
            scale_spectra=cfg.data.scale_spectra,
            save_dir=fold_artifact_dir / "scalers",
        )

        # ── DataLoaders ──────────────────────────────────────────────────
        train_sample_weights = None
        if bool(cfg.training.get("use_sample_weights", True)):
            dataset_col = cfg.data.get("cv_stratify_col", "dataset")
            train_sample_weights = _compute_fold_sample_weights(df, train_idx, dataset_col=dataset_col)
            logger.info(
                f"Fold {fold} sample weights | min={float(train_sample_weights.min()):.4f} "
                f"mean={float(train_sample_weights.mean()):.4f} max={float(train_sample_weights.max()):.4f}"
            )

        train_loader, val_loader = make_loaders(
            train_spectra=spectra[train_idx],
            train_targets=targets[train_idx],
            val_spectra=spectra[val_idx],
            val_targets=targets[val_idx],
            spectra_scaler=spectra_scaler,
            target_scaler=target_scaler,
            transform_targets_in_loader=False,
            train_sample_weights=train_sample_weights,
            train_augmentation=bool(cfg.training.augmentation.enabled),
            aug_prob=float(cfg.training.augmentation.aug_prob),
            betashift=float(cfg.training.augmentation.betashift),
            slopeshift=float(cfg.training.augmentation.slopeshift),
            multishift=float(cfg.training.augmentation.multishift),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )

        # ── Model + loss ─────────────────────────────────────────────────
        model = build_model(cfg)
        criterion = build_loss(cfg)
        logger.info(f"Model parameters: {model.count_parameters():,}")

        if wandb.run:
            wandb.watch(model, criterion, log="gradients", log_freq=100)

        # ── Train ────────────────────────────────────────────────────────
        ckpt_dir = ckpt_root
        trainer = Trainer(
            model=model,
            criterion=criterion,
            cfg=cfg,
            target_transformer=target_scaler,
            fold=fold,
            checkpoint_dir=ckpt_dir,
            device=str(device),
        )
        history = trainer.fit(train_loader, val_loader)

        # Save training curves
        pd.DataFrame(history).to_csv(fold_artifact_dir / "history.csv", index=False)

        # ── Evaluate best model on val set ───────────────────────────────
        trainer.load_best_weights()
        metrics_df = evaluate(
            model=trainer.model,
            spectra=spectra[val_idx],
            targets=targets[val_idx],
            spectra_scaler=spectra_scaler,
            target_scaler=target_scaler,
            trait_names=trait_cols,
            device=str(device),
            output_dir=fold_artifact_dir / "eval",
            prefix=f"fold{fold}_val",
        )

        logger.info(f"\nFold {fold} validation metrics:\n{metrics_df.to_string(index=False)}")
        fold_metrics.append(metrics_df)

        if wandb.run:
            wandb.log(metrics_to_wandb(metrics_df, prefix=f"fold{fold}/val_final"))

        finish_wandb()

    # ── Aggregate across folds ───────────────────────────────────────────
    _aggregate_cv_results(fold_metrics, trait_cols, output_root, cfg, logger)


def _aggregate_cv_results(fold_metrics, trait_cols, output_root, cfg, logger):
    """Compute mean ± std across folds and log to W&B summary run."""
    all_dfs = pd.concat(
        [df.assign(fold=i) for i, df in enumerate(fold_metrics)],
        ignore_index=True,
    )
    all_dfs.to_csv(output_root / "all_fold_metrics.csv", index=False)

    summary = (
        all_dfs[all_dfs["trait"] != "MEAN"]
        .groupby("trait")[["rmse", "mae", "r2", "nrmse", "bias"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(output_root / "cv_summary.csv")
    logger.info(f"\n{'='*60}\nCV Summary (mean ± std across {cfg.data.n_folds} folds):\n"
                f"{summary.to_string()}\n{'='*60}")

    # One final W&B summary run
    if cfg.logging.wandb.enabled:
        init_wandb(cfg, run_name=f"{cfg.project.name}_cv_summary")
        flat = {}
        for trait in trait_cols:
            row = summary.loc[trait]
            for metric in ["rmse", "mae", "r2"]:
                flat[f"cv/{trait}/{metric}_mean"] = row[(metric, "mean")]
                flat[f"cv/{trait}/{metric}_std"] = row[(metric, "std")]
        wandb.log(flat)
        finish_wandb()


if __name__ == "__main__":
    main()
