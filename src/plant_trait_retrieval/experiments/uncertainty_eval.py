from __future__ import annotations

import copy
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PowerTransformer

from ..data.loaders import make_loaders
from ..data.preprocessing import build_transforms
from ..data.splitter import TransferSplitter
from ..evaluation.evaluator import predict
from ..evaluation.metrics import compute_metrics
from ..models.registry import build_model
from ..training.losses import build_loss
from ..training.trainer import Trainer
from ..uncertainty.distance import disun_distance_features
from ..utils.io import load_dataset
from ..utils.misc import get_device, get_logger, seed_everything


def _normalize_stage(stage: str) -> str:
    s = str(stage).strip().lower()
    alias = {
        "full": "all",
        "end2end": "all",
        "transferability": "transfer",
        "dist": "distance",
    }
    s = alias.get(s, s)
    if s not in {"all", "transfer", "distance"}:
        raise ValueError(f"Invalid experiment.stage='{stage}'. Use one of: all, transfer, distance.")
    return s


def _fill_median(train_arr: np.ndarray, test_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(train_arr).astype(np.float32), imp.transform(test_arr).astype(np.float32)


def _fit_plsr(train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray, n_comp: int = 30) -> np.ndarray:
    preds = np.full((len(pred_x), train_y.shape[1]), np.nan, dtype=np.float32)
    for t in range(train_y.shape[1]):
        yt = train_y[:, t]
        finite = np.isfinite(yt)
        if finite.sum() < 10:
            continue
        n = max(2, min(n_comp, finite.sum() - 1, train_x.shape[1] - 1))
        m = PLSRegression(n_components=n)
        m.fit(train_x[finite], yt[finite])
        preds[:, t] = m.predict(pred_x).reshape(-1)
    return preds


@torch.no_grad()
def _extract_embeddings(
    model,
    spectra: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if not hasattr(model, "extract_features"):
        raise ValueError("Model must implement extract_features() for embedding-based Dis_UN.")
    model = model.to(device).eval()
    x = torch.from_numpy(spectra.astype(np.float32))
    embs = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].unsqueeze(1).to(device)
        feat = model.extract_features(xb)
        embs.append(feat.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


def _fit_quantile_models(
    x_calib: pd.DataFrame,
    abs_resid_calib: np.ndarray,
    quantile: float,
    alpha: float,
    transform_x: bool,
    transform_y: bool,
    x_method: str,
    y_method: str,
) -> list[dict | None]:
    models: list[dict | None] = []
    for t in range(abs_resid_calib.shape[1]):
        y = abs_resid_calib[:, t]
        finite = np.isfinite(y) & np.isfinite(x_calib).all(axis=1).values
        if finite.sum() < 30:
            models.append(None)
            continue
        x_fit = x_calib.loc[finite].to_numpy(dtype=np.float64)
        y_fit = y[finite].astype(np.float64).reshape(-1, 1)

        x_scaler = None
        if transform_x:
            x_scaler = PowerTransformer(method=x_method)
            x_fit = x_scaler.fit_transform(x_fit)

        y_scaler = None
        if transform_y:
            y_method_eff = y_method
            if y_method_eff == "box-cox" and np.any(y_fit <= 0):
                y_method_eff = "yeo-johnson"
            y_scaler = PowerTransformer(method=y_method_eff)
            y_fit = y_scaler.fit_transform(y_fit)

        reg = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
        reg.fit(x_fit, y_fit.reshape(-1))
        models.append(
            {
                "model": reg,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
            }
        )
    return models


def _predict_quantile_models(models: list[dict | QuantileRegressor | None], x_new: pd.DataFrame) -> np.ndarray:
    preds = np.full((len(x_new), len(models)), np.nan, dtype=np.float32)
    x_raw_all = x_new.to_numpy(dtype=np.float64)
    for t, model in enumerate(models):
        if model is None:
            continue
        finite = np.isfinite(x_raw_all).all(axis=1)
        if finite.sum() == 0:
            continue
        x_fit = x_raw_all[finite]

        if isinstance(model, dict):
            reg = model["model"]
            x_scaler = model.get("x_scaler", None)
            y_scaler = model.get("y_scaler", None)
            if x_scaler is not None:
                x_fit = x_scaler.transform(x_fit)
            p = reg.predict(x_fit).reshape(-1, 1)
            if y_scaler is not None:
                p = y_scaler.inverse_transform(p)
            p = p.reshape(-1)
        else:
            p = model.predict(x_fit)

        out = np.full(len(x_new), np.nan, dtype=np.float32)
        out[finite] = np.clip(p.astype(np.float32), a_min=0.0, a_max=None)
        preds[:, t] = out
    return preds


@hydra.main(config_path="../../../configs/experiments", config_name="uncertainty", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(int(cfg.project.seed))

    spectra, targets, _, trait_cols, df = load_dataset(cfg)
    # Keep strict row filtering aligned with train_cv.py.
    valid_target_mask = np.isfinite(targets).any(axis=1)
    valid_spectra_mask = np.isfinite(spectra).all(axis=1)
    valid_row_mask = valid_target_mask & valid_spectra_mask
    dropped_bad_targets = int((~valid_target_mask).sum())
    dropped_bad_spectra = int((~valid_spectra_mask).sum())
    dropped_rows = int((~valid_row_mask).sum())
    if dropped_rows > 0:
        spectra = spectra[valid_row_mask]
        targets = targets[valid_row_mask]
        df = df.loc[valid_row_mask].reset_index(drop=True)
        logger.warning(
            "Dropped unusable rows before uncertainty split: "
            f"all-NaN/non-finite targets={dropped_bad_targets}, "
            f"non-finite spectra={dropped_bad_spectra}, total_dropped={dropped_rows}."
        )
    logger.info(
        f"Dataset: {spectra.shape[0]} samples | {spectra.shape[1]} bands | {targets.shape[1]} traits"
    )

    output_root = Path(cfg.experiment.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = _normalize_stage(cfg.experiment.get("stage", "all"))
    logger.info(f"Uncertainty pipeline stage: {stage}")

    # Dis_UN follows transferability (leave-one-domain-out), not random KFold.
    domain_col = str(cfg.experiment.transfer.domain_col)
    if domain_col not in df.columns:
        raise ValueError(f"Missing transfer domain column '{domain_col}' in dataset.")
    domains = sorted(df[domain_col].dropna().astype(str).unique().tolist())
    if not bool(cfg.experiment.transfer.run_all_domains):
        domains = [str(cfg.experiment.transfer.target_domain)]

    all_summary_rows: list[pd.DataFrame] = []

    for run_idx, target_domain in enumerate(domains):
        splitter = TransferSplitter(domain_col=domain_col, target_domain=target_domain)
        source_idx, target_idx = splitter.split(df.assign(**{domain_col: df[domain_col].astype(str)}))
        if len(source_idx) < 100 or len(target_idx) < 10:
            logger.warning(
                f"Skipping target_domain={target_domain} because source={len(source_idx)} target={len(target_idx)}"
            )
            continue

        rng = np.random.default_rng(int(cfg.project.seed) + run_idx)
        # Permute positions, then map back to global source indices.
        perm = rng.permutation(len(source_idx))
        n_cal = max(64, int(float(cfg.experiment.transfer.calibration_fraction) * len(source_idx)))
        n_cal = min(n_cal, len(source_idx) - 1)
        calib_idx = source_idx[perm[:n_cal]]
        train_idx = source_idx[perm[n_cal:]]

        out_dir = output_root / f"target_{target_domain}"
        out_dir.mkdir(parents=True, exist_ok=True)
        transfer_artifact_path = out_dir / "transfer_stage_artifacts.npz"

        if stage in {"all", "transfer"}:
            x_train_raw, x_calib_raw = _fill_median(spectra[train_idx], spectra[calib_idx])
            _, x_target_raw = _fill_median(spectra[train_idx], spectra[target_idx])
            y_train = targets[train_idx]
            y_calib = targets[calib_idx]
            y_target = targets[target_idx]

            base_model = str(cfg.experiment.base_model)
            if base_model == "plsr":
                pred_calib = _fit_plsr(x_train_raw, y_train, x_calib_raw, n_comp=int(cfg.experiment.plsr_components))
                pred_target = _fit_plsr(x_train_raw, y_train, x_target_raw, n_comp=int(cfg.experiment.plsr_components))
                # For PLSR we use spectral space for both branches.
                emb_train = x_train_raw
                emb_calib = x_calib_raw
                emb_target = x_target_raw
                x_train_for_dist = x_train_raw
                x_calib_for_dist = x_calib_raw
                x_target_for_dist = x_target_raw
            else:
                tcfg = copy.deepcopy(cfg)
                tcfg.model.name = base_model
                tcfg.logging.wandb.enabled = False

                spectra_scaler, target_scaler = build_transforms(
                    train_spectra=x_train_raw,
                    train_targets=y_train,
                    power_method=tcfg.data.power_method,
                    scale_spectra=bool(tcfg.data.scale_spectra),
                    save_dir=out_dir / "scalers",
                )
                train_loader, val_loader = make_loaders(
                    train_spectra=x_train_raw,
                    train_targets=y_train,
                    val_spectra=x_calib_raw,
                    val_targets=y_calib,
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
                    fold=run_idx,
                    checkpoint_dir=Path("checkpoints/uncertainty") / base_model / target_domain,
                    device=str(get_device()),
                )
                trainer.fit(train_loader, val_loader)
                trainer.load_best_weights()

                pred_calib_t = predict(trainer.model, x_calib_raw, spectra_scaler, device=str(get_device()))
                pred_target_t = predict(trainer.model, x_target_raw, spectra_scaler, device=str(get_device()))
                pred_calib = target_scaler.inverse_transform(pred_calib_t)
                pred_target = target_scaler.inverse_transform(pred_target_t)

                x_train_for_dist = spectra_scaler.transform(x_train_raw) if spectra_scaler is not None else x_train_raw
                x_calib_for_dist = spectra_scaler.transform(x_calib_raw) if spectra_scaler is not None else x_calib_raw
                x_target_for_dist = spectra_scaler.transform(x_target_raw) if spectra_scaler is not None else x_target_raw

                emb_train = _extract_embeddings(
                    trainer.model,
                    x_train_for_dist,
                    batch_size=int(cfg.experiment.distance.embedding_batch_size),
                    device=str(get_device()),
                )
                emb_calib = _extract_embeddings(
                    trainer.model,
                    x_calib_for_dist,
                    batch_size=int(cfg.experiment.distance.embedding_batch_size),
                    device=str(get_device()),
                )
                emb_target = _extract_embeddings(
                    trainer.model,
                    x_target_for_dist,
                    batch_size=int(cfg.experiment.distance.embedding_batch_size),
                    device=str(get_device()),
                )

            np.savez_compressed(
                transfer_artifact_path,
                train_idx=train_idx.astype(np.int64),
                calib_idx=calib_idx.astype(np.int64),
                target_idx=target_idx.astype(np.int64),
                y_calib=y_calib.astype(np.float32),
                y_target=y_target.astype(np.float32),
                pred_calib=pred_calib.astype(np.float32),
                pred_target=pred_target.astype(np.float32),
                x_train_for_dist=x_train_for_dist.astype(np.float32),
                x_calib_for_dist=x_calib_for_dist.astype(np.float32),
                x_target_for_dist=x_target_for_dist.astype(np.float32),
                emb_train=emb_train.astype(np.float32),
                emb_calib=emb_calib.astype(np.float32),
                emb_target=emb_target.astype(np.float32),
            )
            logger.info(f"Saved transfer-stage artifacts for target={target_domain} to {transfer_artifact_path}")

            if stage == "transfer":
                continue
        else:
            if not transfer_artifact_path.exists():
                raise FileNotFoundError(
                    f"Missing transfer artifacts for target={target_domain}: {transfer_artifact_path}. "
                    "Run with experiment.stage=transfer first."
                )
            art = np.load(transfer_artifact_path)
            y_calib = art["y_calib"]
            y_target = art["y_target"]
            pred_calib = art["pred_calib"]
            pred_target = art["pred_target"]
            x_train_for_dist = art["x_train_for_dist"]
            x_calib_for_dist = art["x_calib_for_dist"]
            x_target_for_dist = art["x_target_for_dist"]
            emb_train = art["emb_train"]
            emb_calib = art["emb_calib"]
            emb_target = art["emb_target"]
            target_idx = art["target_idx"].astype(np.int64)
            logger.info(f"Loaded transfer-stage artifacts for target={target_domain} from {transfer_artifact_path}")

        # Distance features (Dis_UN): embedding + spectral, cosine, normalized, q50.
        calib_feats = disun_distance_features(
            train_embed=emb_train,
            query_embed=emb_calib,
            train_spectra=x_train_for_dist,
            query_spectra=x_calib_for_dist,
            n_neighbors=int(cfg.experiment.distance.n_neighbors),
            quantile=float(cfg.experiment.distance.quantile),
            normalize_vectors=bool(cfg.experiment.distance.normalize_vectors),
            normalize_by_train=bool(cfg.experiment.distance.normalize_by_train),
            use_average=bool(cfg.experiment.distance.use_average),
            gpu=int(cfg.experiment.distance.gpu),
            require_faiss=bool(cfg.experiment.distance.require_faiss),
        )
        target_feats = disun_distance_features(
            train_embed=emb_train,
            query_embed=emb_target,
            train_spectra=x_train_for_dist,
            query_spectra=x_target_for_dist,
            n_neighbors=int(cfg.experiment.distance.n_neighbors),
            quantile=float(cfg.experiment.distance.quantile),
            normalize_vectors=bool(cfg.experiment.distance.normalize_vectors),
            normalize_by_train=bool(cfg.experiment.distance.normalize_by_train),
            use_average=bool(cfg.experiment.distance.use_average),
            gpu=int(cfg.experiment.distance.gpu),
            require_faiss=bool(cfg.experiment.distance.require_faiss),
        )
        calib_df = pd.DataFrame(calib_feats)
        target_df = pd.DataFrame(target_feats)
        np.savez_compressed(
            out_dir / "distance_reference.npz",
            train_embed=emb_train.astype(np.float32),
            train_spectra=x_train_for_dist.astype(np.float32),
        )

        predictor_subset = list(cfg.experiment.disun.predictor_subset)
        missing_cal = [c for c in predictor_subset if c not in calib_df.columns]
        if missing_cal:
            raise ValueError(f"Missing required Dis_UN predictors: {missing_cal}")

        abs_resid_calib = np.abs(pred_calib - y_calib)
        quant_models = _fit_quantile_models(
            x_calib=calib_df[predictor_subset],
            abs_resid_calib=abs_resid_calib,
            quantile=float(cfg.experiment.disun.quantile),
            alpha=float(cfg.experiment.disun.alpha),
            transform_x=bool(cfg.experiment.disun.transform_x),
            transform_y=bool(cfg.experiment.disun.transform_y),
            x_method=str(cfg.experiment.disun.x_transform_method),
            y_method=str(cfg.experiment.disun.y_transform_method),
        )
        un_target = _predict_quantile_models(quant_models, target_df[predictor_subset])
        with open(out_dir / "quantile_models.pkl", "wb") as f:
            pickle.dump(
                {
                    "models": quant_models,
                    "predictor_subset": predictor_subset,
                    "quantile": float(cfg.experiment.disun.quantile),
                    "alpha": float(cfg.experiment.disun.alpha),
                    "transform_x": bool(cfg.experiment.disun.transform_x),
                    "transform_y": bool(cfg.experiment.disun.transform_y),
                    "x_transform_method": str(cfg.experiment.disun.x_transform_method),
                    "y_transform_method": str(cfg.experiment.disun.y_transform_method),
                },
                f,
            )

        # Evaluation summary.
        metrics_df = compute_metrics(y_true=y_target, y_pred=pred_target, trait_names=trait_cols)
        abs_err_target = np.nanmean(np.abs(pred_target - y_target), axis=1)
        un_target_mean = np.nanmean(un_target, axis=1)
        corr = float(
            np.corrcoef(
                np.nan_to_num(un_target_mean, nan=0.0),
                np.nan_to_num(abs_err_target, nan=0.0),
            )[0, 1]
        )
        mean_row = metrics_df[metrics_df["trait"] == "MEAN"].copy()
        mean_row["target_domain"] = target_domain
        mean_row["uncertainty_abs_error_corr"] = corr
        all_summary_rows.append(mean_row)

        pred_df = pd.DataFrame(pred_target, columns=trait_cols)
        un_df = pd.DataFrame(un_target, columns=[f"un_{t}" for t in trait_cols])
        out_df = pd.concat([pred_df, un_df], axis=1)
        out_df["uncertainty_mean"] = un_target_mean
        out_df["abs_error_mean"] = abs_err_target
        out_df[domain_col] = target_domain
        if "lat" in df.columns and "lon" in df.columns:
            out_df["lat"] = df.iloc[target_idx]["lat"].to_numpy()
            out_df["lon"] = df.iloc[target_idx]["lon"].to_numpy()

        calib_df.to_csv(out_dir / "dist_features_calibration.csv", index=False)
        target_df.to_csv(out_dir / "dist_features_target.csv", index=False)
        metrics_df.to_csv(out_dir / "prediction_metrics_target.csv", index=False)
        out_df.to_csv(out_dir / "predictions_with_disun_uncertainty.csv", index=False)

        logger.info(
            f"Dis_UN target={target_domain} | samples={len(target_idx)} | "
            f"uncertainty-abs_error corr={corr:.4f}"
        )

    if all_summary_rows:
        summary = pd.concat(all_summary_rows, ignore_index=True)
        summary.to_csv(output_root / "uncertainty_transfer_summary.csv", index=False)
        logger.info(f"Saved Dis_UN transfer summary to {output_root / 'uncertainty_transfer_summary.csv'}")


if __name__ == "__main__":
    main()
