from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from ..data.loaders import make_loaders
from ..data.preprocessing import build_transforms
from ..evaluation.evaluator import predict
from ..evaluation.metrics import compute_metrics
from ..models.registry import build_model
from ..training.losses import build_loss
from ..training.trainer import Trainer
from ..utils.io import load_dataset
from ..utils.misc import get_device, get_logger, seed_everything


@dataclass
class TrialResult:
    params: dict[str, Any]
    score: float


def _fill_spectra_with_train_median(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    train_f = imp.fit_transform(train_x)
    test_f = imp.transform(test_x)
    return train_f.astype(np.float32), test_f.astype(np.float32)


def _fill_targets_with_train_median(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    train_f = imp.fit_transform(train_y)
    test_f = imp.transform(test_y)
    return train_f.astype(np.float32), test_f.astype(np.float32)


def _tune_efficientnet(
    cfg: DictConfig,
    spectra: np.ndarray,
    targets: np.ndarray,
    n_trials: int,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    logger = get_logger(__name__)
    rng = np.random.default_rng(seed)

    trial_space = cfg.experiment.tuning.search_space
    trials: list[dict[str, Any]] = []
    for _ in range(n_trials):
        trials.append(
            {
                "learning_rate": float(rng.choice(list(trial_space.learning_rate))),
                "weight_decay": float(rng.choice(list(trial_space.weight_decay))),
                "dropout_rate": float(rng.choice(list(trial_space.dropout_rate))),
                "delta": float(rng.choice(list(trial_space.huber_delta))),
                "batch_size": int(rng.choice(list(trial_space.batch_size))),
            }
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    trial_results: list[TrialResult] = []

    for idx, params in enumerate(trials):
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(spectra)):
            tcfg = copy.deepcopy(cfg)
            tcfg.model.name = "efficientnet1d_b0"
            tcfg.model.dropout_rate = params["dropout_rate"]
            tcfg.training.learning_rate = params["learning_rate"]
            tcfg.training.weight_decay = params["weight_decay"]
            tcfg.training.loss.delta = params["delta"]
            tcfg.training.batch_size = params["batch_size"]
            tcfg.training.epochs = int(cfg.experiment.tuning.max_epochs_per_trial)
            tcfg.training.patience = int(cfg.experiment.tuning.patience_per_trial)
            tcfg.logging.wandb.enabled = False

            tr_x, va_x = _fill_spectra_with_train_median(spectra[tr_idx], spectra[va_idx])
            tr_y, va_y = targets[tr_idx].copy(), targets[va_idx].copy()
            spectra_scaler, target_scaler = build_transforms(
                train_spectra=tr_x,
                train_targets=tr_y,
                power_method=tcfg.data.power_method,
                scale_spectra=bool(tcfg.data.scale_spectra),
                save_dir=None,
            )
            train_loader, val_loader = make_loaders(
                train_spectra=tr_x,
                train_targets=tr_y,
                val_spectra=va_x,
                val_targets=va_y,
                spectra_scaler=spectra_scaler,
                target_scaler=target_scaler,
                transform_targets_in_loader=False,
                batch_size=int(tcfg.training.batch_size),
                num_workers=int(tcfg.training.num_workers),
                pin_memory=bool(tcfg.training.pin_memory),
                train_augmentation=bool(tcfg.training.augmentation.enabled),
                aug_prob=float(tcfg.training.augmentation.aug_prob),
                betashift=float(tcfg.training.augmentation.betashift),
                slopeshift=float(tcfg.training.augmentation.slopeshift),
                multishift=float(tcfg.training.augmentation.multishift),
            )

            model = build_model(tcfg)
            criterion = build_loss(tcfg)
            trainer = Trainer(
                model=model,
                criterion=criterion,
                cfg=tcfg,
                target_transformer=target_scaler,
                fold=fold,
                checkpoint_dir=Path("checkpoints/tuning"),
                device=str(get_device()),
            )
            trainer.fit(train_loader, val_loader)
            trainer.load_best_weights()
            preds_t = predict(
                trainer.model,
                spectra=va_x,
                spectra_scaler=spectra_scaler,
                device=str(get_device()),
            )
            preds = target_scaler.inverse_transform(preds_t)
            y_true = target_scaler.inverse_transform(target_scaler.transform(va_y))
            mdf = compute_metrics(y_true=y_true, y_pred=preds)
            score = float(mdf[mdf["trait"] == "MEAN"]["rmse"].iloc[0])
            fold_scores.append(score)

        trial_score = float(np.nanmean(fold_scores))
        trial_results.append(TrialResult(params=params, score=trial_score))
        logger.info(f"Tuning trial {idx + 1}/{len(trials)} params={params} mean_rmse={trial_score:.6f}")

    best = sorted(trial_results, key=lambda x: x.score)[0]
    logger.info(f"Best tuning params: {best.params} with mean_rmse={best.score:.6f}")
    return best.params


def _fit_plsr_per_trait(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    max_components: int,
) -> np.ndarray:
    preds = np.full((len(val_x), train_y.shape[1]), np.nan, dtype=np.float32)
    for t in range(train_y.shape[1]):
        y_t = train_y[:, t]
        finite = np.isfinite(y_t)
        if finite.sum() < 10:
            continue
        x_t = train_x[finite]
        y_t = y_t[finite]
        n_comp = max(2, min(max_components, x_t.shape[0] - 1, x_t.shape[1] - 1))
        plsr = PLSRegression(n_components=n_comp, scale=True)
        plsr.fit(x_t, y_t)
        preds[:, t] = plsr.predict(val_x).reshape(-1).astype(np.float32)
    return preds


def _run_single_fold_deep(
    cfg: DictConfig,
    model_name: str,
    variant: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    spectra: np.ndarray,
    targets: np.ndarray,
    trait_cols: list[str],
    output_root: Path,
) -> pd.DataFrame:
    fold_dir = output_root / model_name / variant / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tr_x_raw, va_x_raw = spectra[train_idx], spectra[val_idx]
    tr_y_raw, va_y_raw = targets[train_idx], targets[val_idx]

    tr_x, va_x = _fill_spectra_with_train_median(tr_x_raw, va_x_raw)
    tr_y, va_y = tr_y_raw.copy(), va_y_raw.copy()
    if variant == "filled":
        tr_y, va_y = _fill_targets_with_train_median(tr_y, va_y)

    spectra_scaler, target_scaler = build_transforms(
        train_spectra=tr_x,
        train_targets=tr_y,
        power_method=cfg.data.power_method,
        scale_spectra=bool(cfg.data.scale_spectra),
        save_dir=fold_dir / "scalers",
    )

    train_loader, val_loader = make_loaders(
        train_spectra=tr_x,
        train_targets=tr_y,
        val_spectra=va_x,
        val_targets=va_y,
        spectra_scaler=spectra_scaler,
        target_scaler=target_scaler,
        transform_targets_in_loader=False,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(cfg.training.pin_memory),
        train_augmentation=bool(cfg.training.augmentation.enabled),
        aug_prob=float(cfg.training.augmentation.aug_prob),
        betashift=float(cfg.training.augmentation.betashift),
        slopeshift=float(cfg.training.augmentation.slopeshift),
        multishift=float(cfg.training.augmentation.multishift),
    )

    fcfg = copy.deepcopy(cfg)
    fcfg.model.name = model_name
    fcfg.logging.wandb.enabled = bool(cfg.experiment.get("wandb", False))
    model = build_model(fcfg)
    criterion = build_loss(fcfg)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        cfg=fcfg,
        target_transformer=target_scaler,
        fold=fold,
        checkpoint_dir=Path("checkpoints/baseline") / model_name / variant,
        device=str(get_device()),
    )
    history = trainer.fit(train_loader, val_loader)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    trainer.load_best_weights()

    preds_t = predict(
        trainer.model,
        spectra=va_x,
        spectra_scaler=spectra_scaler,
        device=str(get_device()),
    )
    preds = target_scaler.inverse_transform(preds_t)
    y_true = target_scaler.inverse_transform(target_scaler.transform(va_y_raw))
    mdf = compute_metrics(y_true=y_true, y_pred=preds, trait_names=trait_cols)
    mdf.to_csv(fold_dir / "metrics.csv", index=False)
    return mdf


@hydra.main(config_path="../../../configs/experiments", config_name="baseline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(int(cfg.project.seed))

    spectra, targets, _, trait_cols, _ = load_dataset(cfg)
    logger.info(f"Loaded dataset: X={spectra.shape} y={targets.shape}")

    output_root = Path(cfg.experiment.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if bool(cfg.experiment.tuning.enabled):
        best_params = _tune_efficientnet(
            cfg,
            spectra=spectra,
            targets=targets,
            n_trials=int(cfg.experiment.tuning.n_trials),
            n_folds=int(cfg.experiment.tuning.n_folds),
            seed=int(cfg.project.seed),
        )
        with open(output_root / "best_tuned_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        cfg.training.learning_rate = best_params["learning_rate"]
        cfg.training.weight_decay = best_params["weight_decay"]
        cfg.training.loss.delta = best_params["delta"]
        cfg.training.batch_size = best_params["batch_size"]
        cfg.model.dropout_rate = best_params["dropout_rate"]

    models = list(cfg.experiment.model_variants)
    data_variants = list(cfg.experiment.data_variants)

    kf = KFold(n_splits=int(cfg.data.n_folds), shuffle=True, random_state=int(cfg.project.seed))
    summary_rows: list[pd.DataFrame] = []

    for model_name in models:
        for variant in data_variants:
            logger.info(f"Running baseline model={model_name} data_variant={variant}")
            fold_tables = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(spectra)):
                if model_name == "plsr":
                    fold_dir = output_root / model_name / variant / f"fold_{fold}"
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    tr_x, va_x = _fill_spectra_with_train_median(spectra[train_idx], spectra[val_idx])
                    tr_y, va_y = targets[train_idx].copy(), targets[val_idx].copy()
                    if variant == "filled":
                        tr_y, va_y = _fill_targets_with_train_median(tr_y, va_y)
                    preds = _fit_plsr_per_trait(
                        train_x=tr_x,
                        train_y=tr_y,
                        val_x=va_x,
                        max_components=int(cfg.experiment.plsr.max_components),
                    )
                    mdf = compute_metrics(y_true=targets[val_idx], y_pred=preds, trait_names=trait_cols)
                    mdf.to_csv(fold_dir / "metrics.csv", index=False)
                else:
                    mdf = _run_single_fold_deep(
                        cfg=cfg,
                        model_name=model_name,
                        variant=variant,
                        fold=fold,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        spectra=spectra,
                        targets=targets,
                        trait_cols=trait_cols,
                        output_root=output_root,
                    )
                mean_row = mdf[mdf["trait"] == "MEAN"].copy()
                mean_row["fold"] = fold
                fold_tables.append(mean_row)

            res = pd.concat(fold_tables, ignore_index=True)
            res["model"] = model_name
            res["data_variant"] = variant
            res.to_csv(output_root / model_name / variant / "cv_mean_metrics.csv", index=False)
            summary_rows.append(res)

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        agg = (
            summary.groupby(["model", "data_variant"])[["rmse", "mae", "r2", "nrmse", "bias"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.to_csv(output_root / "all_results_long.csv", index=False)
        agg.to_csv(output_root / "summary_by_model_variant.csv", index=False)
        logger.info("Baseline run complete. Saved all experiment tables.")
        logger.info("\n" + OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
