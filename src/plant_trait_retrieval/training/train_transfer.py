"""Transferability experiment script.

Trains on all source-domain samples, evaluates on the held-out target domain.
Optionally fine-tunes on a small fraction of target data.

Usage
-----
    python -m plant_trait_retrieval.training.train_transfer \
        --config-name transferability \
        data.transferability.target_domain=MyTargetSite
"""
from __future__ import annotations

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
from ..data.splitter import TransferSplitter
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


@hydra.main(config_path="../../../configs", config_name="transferability", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(cfg.project.seed)
    device = get_device()
    print_system_info(logger)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Load data ─────────────────────────────────────────────────────────
    spectra, targets, spectra_cols, trait_cols, df = load_dataset(cfg)
    logger.info(f"Dataset: {spectra.shape[0]} samples | "
                f"{spectra.shape[1]} bands | {targets.shape[1]} traits")

    # ── Transfer split setup ──────────────────────────────────────────────
    tcfg = cfg.data.transferability
    domain_col = tcfg.source_domain_col
    run_all_domains = bool(tcfg.get("run_all_domains", False))
    val_fraction = float(tcfg.get("val_fraction", 0.2))

    if run_all_domains:
        domain_values = sorted(df[domain_col].dropna().astype(str).unique().tolist())
    else:
        domain_values = [str(tcfg.target_domain)]

    all_results: list[pd.DataFrame] = []
    exp_id = cfg.get("experiment_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_root = Path("results/transfer")
    if run_all_domains:
        base_output_root = base_output_root / f"transfer_all_{exp_id}"
    else:
        base_output_root = base_output_root / f"transfer_single_{exp_id}"
    base_output_root.mkdir(parents=True, exist_ok=True)
    data_tag = Path(cfg.data.data_path).stem
    transfer_scope = "all_domains" if run_all_domains else "single_domain"
    ckpt_experiment_root = (
        Path(cfg.logging.checkpoint_dir)
        / "transfer"
        / cfg.project.name
        / data_tag
        / transfer_scope
        / f"models_{exp_id}"
    )
    ckpt_experiment_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Transfer experiment id: {exp_id}")
    logger.info(f"Transfer results root: {base_output_root}")
    logger.info(f"Transfer checkpoint root: {ckpt_experiment_root}")

    for run_idx, target_domain in enumerate(domain_values):
        splitter = TransferSplitter(domain_col=domain_col, target_domain=target_domain)
        source_idx, target_idx = splitter.split(df.assign(**{domain_col: df[domain_col].astype(str)}))
        if len(target_idx) == 0:
            logger.warning(f"Skipping target domain '{target_domain}' (no samples).")
            continue
        if len(source_idx) < 2:
            logger.warning(f"Skipping target domain '{target_domain}' (source too small).")
            continue

        logger.info(
            f"\n{'='*60}\nTransfer run {run_idx+1}/{len(domain_values)} | "
            f"target='{target_domain}' | source={len(source_idx)} | target={len(target_idx)}\n{'='*60}"
        )

        output_root = base_output_root / target_domain
        output_root.mkdir(parents=True, exist_ok=True)

        init_wandb(cfg, run_name=f"{cfg.project.name}_transfer_{target_domain}")

        spectra_scaler, target_scaler = build_transforms(
            train_spectra=spectra[source_idx],
            train_targets=targets[source_idx],
            power_method=cfg.data.power_method,
            scale_spectra=cfg.data.scale_spectra,
            save_dir=output_root / "scalers",
        )

        # Validation split from source (default 20%)
        rng = np.random.default_rng(cfg.project.seed)
        perm = rng.permutation(len(source_idx))
        n_val = max(1, int(val_fraction * len(source_idx)))
        n_val = min(n_val, len(source_idx) - 1)
        val_src_idx = source_idx[perm[:n_val]]
        train_src_idx = source_idx[perm[n_val:]]

        train_loader, val_loader = make_loaders(
            train_spectra=spectra[train_src_idx],
            train_targets=targets[train_src_idx],
            val_spectra=spectra[val_src_idx],
            val_targets=targets[val_src_idx],
            spectra_scaler=spectra_scaler,
            target_scaler=target_scaler,
            transform_targets_in_loader=False,
            train_augmentation=bool(cfg.training.augmentation.enabled),
            aug_prob=float(cfg.training.augmentation.aug_prob),
            betashift=float(cfg.training.augmentation.betashift),
            slopeshift=float(cfg.training.augmentation.slopeshift),
            multishift=float(cfg.training.augmentation.multishift),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )

        model = build_model(cfg)
        criterion = build_loss(cfg)
        logger.info(f"Model parameters: {model.count_parameters():,}")

        if wandb.run:
            wandb.watch(model, criterion, log="gradients", log_freq=100)

        ckpt_dir = ckpt_experiment_root / target_domain
        trainer = Trainer(
            model=model,
            criterion=criterion,
            cfg=cfg,
            target_transformer=target_scaler,
            fold=0,
            checkpoint_dir=ckpt_dir,
            device=str(device),
        )
        history = trainer.fit(train_loader, val_loader)
        pd.DataFrame(history).to_csv(output_root / "source_history.csv", index=False)

        trainer.load_best_weights()
        metrics_df = evaluate(
            model=trainer.model,
            spectra=spectra[target_idx],
            targets=targets[target_idx],
            spectra_scaler=spectra_scaler,
            target_scaler=target_scaler,
            trait_names=trait_cols,
            device=str(device),
            output_dir=output_root / "eval",
            prefix="target_domain",
        )
        metrics_df["target_domain"] = target_domain
        all_results.append(metrics_df)

        logger.info(
            f"\nTarget domain '{target_domain}' metrics:\n"
            f"{metrics_df.to_string(index=False)}"
        )

        if wandb.run:
            wandb.log(metrics_to_wandb(metrics_df.drop(columns=["target_domain"]), prefix="transfer/target"))
            finish_wandb()

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged.to_csv(base_output_root / "all_domains_metrics.csv", index=False)
        mean_rows = merged[merged["trait"] == "MEAN"].copy()
        if not mean_rows.empty:
            summary = mean_rows[["target_domain", "rmse", "mae", "r2", "nrmse", "bias"]]
            summary.to_csv(base_output_root / "all_domains_mean_metrics.csv", index=False)
            logger.info(
                f"\nSaved transferability summary to:\n"
                f"- {base_output_root / 'all_domains_metrics.csv'}\n"
                f"- {base_output_root / 'all_domains_mean_metrics.csv'}"
            )


if __name__ == "__main__":
    main()
