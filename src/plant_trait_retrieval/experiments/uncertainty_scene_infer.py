from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.preprocessing import PowerTransformerWrapper, SpectraStandardScaler
from ..data.hsi_scene import preprocess_scene_tif_to_1522
from ..models.registry import build_model
from ..uncertainty.distance import disun_distance_features
from ..utils.misc import get_device, get_logger


def _predict_quantile_models(models: list[dict | object | None], x_new: pd.DataFrame) -> np.ndarray:
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
        out[finite] = np.clip(np.asarray(p, dtype=np.float32), 0.0, None)
        preds[:, t] = out
    return preds


@torch.no_grad()
def _predict_batched(
    model,
    spectra: np.ndarray,
    batch_size: int,
    device: str,
    logger=None,
) -> np.ndarray:
    model = model.to(device).eval()
    x = torch.from_numpy(spectra.astype(np.float32))
    preds = []
    total = len(x)
    n_batches = int(np.ceil(total / max(1, batch_size)))
    for b, i in enumerate(range(0, total, batch_size), start=1):
        xb = x[i : i + batch_size].unsqueeze(1).to(device)
        out = model(xb)
        preds.append(out.detach().cpu().numpy().astype(np.float32))
        if logger is not None and (b == 1 or b % 10 == 0 or b == n_batches):
            logger.info(f"[predict] batch {b}/{n_batches} ({min(i + batch_size, total)}/{total} samples)")
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def _extract_embeddings(
    model,
    spectra: np.ndarray,
    batch_size: int,
    device: str,
    logger=None,
) -> np.ndarray:
    if not hasattr(model, "extract_features"):
        raise ValueError("Model must expose extract_features()")
    model = model.to(device).eval()
    x = torch.from_numpy(spectra.astype(np.float32))
    embs = []
    total = len(x)
    n_batches = int(np.ceil(total / max(1, batch_size)))
    for b, i in enumerate(range(0, total, batch_size), start=1):
        xb = x[i : i + batch_size].unsqueeze(1).to(device)
        feat = model.extract_features(xb)
        embs.append(feat.detach().cpu().numpy().astype(np.float32))
        if logger is not None and (b == 1 or b % 10 == 0 or b == n_batches):
            logger.info(f"[embedding] batch {b}/{n_batches} ({min(i + batch_size, total)}/{total} samples)")
    return np.concatenate(embs, axis=0)


def _resolve_spectra_cols(df: pd.DataFrame, n_bands: int) -> list[str]:
    wl_cols = [c for c in df.columns if c.isdigit()]
    if len(wl_cols) >= n_bands:
        wl_cols = sorted(wl_cols, key=lambda c: int(c))
        return wl_cols[:n_bands]
    return df.columns[:n_bands].tolist()


def _write_multiband_tif(
    path: Path,
    data_bands_h_w: np.ndarray,
    transform,
    crs,
    logger=None,
) -> Path:
    try:
        import rasterio
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "rasterio is required to export GeoTIFF outputs. "
            "Install with: poetry add rasterio"
        ) from exc

    nodata = -9999.0
    arr = np.asarray(data_bands_h_w, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(arr)
    if logger is not None:
        logger.info(f"Saved GeoTIFF: {path} | shape={arr.shape}")
    return path


def run_inference(cfg: DictConfig) -> Path:
    logger = get_logger(__name__)
    out_path = Path(cfg.inference.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting scene inference -> output: {out_path}")

    input_type = str(cfg.inference.get("input_type", "csv")).lower()
    if input_type not in {"csv", "tif"}:
        raise ValueError("inference.input_type must be 'csv' or 'tif'.")
    logger.info(f"Input type: {input_type}")

    model_artifact = torch.load(cfg.inference.model_artifact, map_location="cpu")
    artifact_cfg = OmegaConf.create(model_artifact["cfg"])
    model = build_model(artifact_cfg)
    model.load_state_dict(model_artifact["model_state_dict"])
    device = str(get_device())
    model = model.to(device).eval()
    logger.info(f"Loaded model: {cfg.inference.model_artifact} | device={device}")

    spectra_scaler_path = Path(cfg.inference.scalers_dir) / "spectra_scaler.pkl"
    if spectra_scaler_path.exists():
        spectra_scaler = SpectraStandardScaler.load(spectra_scaler_path)
    else:
        spectra_scaler = None
        logger.warning(
            f"spectra_scaler.pkl not found at {spectra_scaler_path}. "
            "Proceeding without spectra scaling."
        )
    target_scaler = PowerTransformerWrapper.load(Path(cfg.inference.scalers_dir) / "target_scaler.pkl")
    logger.info(f"Loaded target scaler from: {Path(cfg.inference.scalers_dir) / 'target_scaler.pkl'}")

    if input_type == "csv":
        scene_df = pd.read_csv(cfg.inference.scene_csv)
        n_bands = int(artifact_cfg.data.n_spectra_bands)
        spectra_cols = _resolve_spectra_cols(scene_df, n_bands=n_bands)
        x_raw = scene_df[spectra_cols].to_numpy(dtype=np.float32)
        x_scaled = spectra_scaler.transform(x_raw) if spectra_scaler is not None else x_raw
        base_out_df = scene_df.reset_index(drop=True).copy()
        valid_mask = np.ones(len(base_out_df), dtype=bool)
        logger.info(f"Loaded CSV scene: {cfg.inference.scene_csv} | samples={len(scene_df)} | bands={len(spectra_cols)}")
    else:
        logger.info(f"Loading TIFF scene: {cfg.inference.scene_tif}")
        logger.info(f"Loading sensor bands: {cfg.inference.sensor_bands_csv}")
        bundle = preprocess_scene_tif_to_1522(
            tif_path=Path(cfg.inference.scene_tif),
            bands_csv=Path(cfg.inference.sensor_bands_csv),
            divide_by=float(cfg.inference.preprocessing.divide_by),
            corrupted_pixel_offset=float(cfg.inference.preprocessing.corrupted_pixel_offset),
            inval_1522=list(cfg.inference.preprocessing.inval_1522),
            polyorder=int(cfg.inference.preprocessing.polyorder),
            deriv=bool(cfg.inference.preprocessing.deriv),
            window=int(cfg.inference.preprocessing.window),
        )
        x_scaled = (
            spectra_scaler.transform(bundle.spectra_1522.astype(np.float32))
            if spectra_scaler is not None
            else bundle.spectra_1522.astype(np.float32)
        )
        base_out_df = bundle.meta_df.reset_index(drop=True).copy()
        valid_mask = base_out_df["is_valid_pixel"].to_numpy(dtype=bool)
        logger.info(
            f"TIFF prepared -> total_pixels={len(base_out_df)} | valid_pixels={int(valid_mask.sum())} | "
            f"model_bands={bundle.spectra_1522.shape[1]}"
        )

    pred_t = _predict_batched(
        model=model,
        spectra=x_scaled,
        batch_size=int(cfg.inference.get("pred_batch_size", 512)),
        device=device,
        logger=logger,
    )
    pred = target_scaler.inverse_transform(pred_t)
    logger.info(f"Prediction complete: shape={pred.shape}")

    with open(cfg.inference.quantile_models_path, "rb") as f:
        quant_blob = pickle.load(f)
    if isinstance(quant_blob, dict) and "models" in quant_blob:
        quant_models = quant_blob["models"]
    else:
        quant_models = quant_blob
    logger.info(f"Loaded quantile models: {cfg.inference.quantile_models_path}")

    ref = np.load(cfg.inference.distance_reference_npz)
    train_embed = ref["train_embed"].astype(np.float32)
    train_spectra = ref["train_spectra"].astype(np.float32)
    logger.info(
        f"Loaded distance reference: {cfg.inference.distance_reference_npz} | "
        f"train_embed={train_embed.shape} train_spectra={train_spectra.shape}"
    )

    emb_scene = _extract_embeddings(
        model,
        x_scaled,
        batch_size=int(cfg.inference.distance.embedding_batch_size),
        device=device,
        logger=logger,
    )
    logger.info(f"Embedding extraction complete: shape={emb_scene.shape}")

    feats = disun_distance_features(
        train_embed=train_embed,
        query_embed=emb_scene,
        train_spectra=train_spectra,
        query_spectra=x_scaled,
        n_neighbors=int(cfg.inference.distance.n_neighbors),
        quantile=float(cfg.inference.distance.quantile),
        normalize_vectors=bool(cfg.inference.distance.normalize_vectors),
        normalize_by_train=bool(cfg.inference.distance.normalize_by_train),
        use_average=bool(cfg.inference.distance.use_average),
        gpu=int(cfg.inference.distance.gpu),
        require_faiss=bool(cfg.inference.distance.require_faiss),
    )
    feat_df = pd.DataFrame(feats)
    logger.info(f"Distance features computed: shape={feat_df.shape}")

    predictor_subset = list(cfg.inference.predictor_subset)
    miss = [c for c in predictor_subset if c not in feat_df.columns]
    if miss:
        raise ValueError(f"Missing predictors for quantile models: {miss}")

    un_pred = _predict_quantile_models(quant_models, feat_df[predictor_subset])
    logger.info(f"Uncertainty prediction complete: shape={un_pred.shape}")

    trait_names = model_artifact.get("trait_names", [f"trait_{i}" for i in range(pred.shape[1])])
    pred_tif_path = None
    unc_tif_path = None
    if input_type == "csv":
        pred_df = pd.DataFrame(pred, columns=[f"pred_{t}" for t in trait_names])
        un_df = pd.DataFrame(un_pred, columns=[f"un_{t}" for t in trait_names])
        out_df = pd.concat([base_out_df, pred_df, un_df], axis=1)
        out_df["uncertainty_mean"] = np.nanmean(un_pred, axis=1)
    else:
        out_df = base_out_df.copy()
        h = int(bundle.n_rows)
        w = int(bundle.n_cols)
        n_traits = int(pred.shape[1])
        pred_cube = np.full((n_traits, h, w), np.nan, dtype=np.float32)
        unc_cube = np.full((n_traits, h, w), np.nan, dtype=np.float32)
        rr = out_df.loc[valid_mask, "row"].to_numpy(dtype=np.int64)
        cc = out_df.loc[valid_mask, "col"].to_numpy(dtype=np.int64)
        for j, t in enumerate(trait_names):
            full_pred = np.full(len(out_df), np.nan, dtype=np.float32)
            full_un = np.full(len(out_df), np.nan, dtype=np.float32)
            full_pred[valid_mask] = pred[:, j]
            full_un[valid_mask] = un_pred[:, j]
            out_df[f"pred_{t}"] = full_pred
            out_df[f"un_{t}"] = full_un
            pred_cube[j, rr, cc] = pred[:, j]
            unc_cube[j, rr, cc] = un_pred[:, j]
        out_df["uncertainty_mean"] = np.nanmean(
            out_df[[f"un_{t}" for t in trait_names]].to_numpy(dtype=np.float32),
            axis=1,
        )
        out_df["scene_height"] = int(bundle.n_rows)
        out_df["scene_width"] = int(bundle.n_cols)

        pred_tif_cfg = cfg.inference.get("output_pred_tif", None)
        unc_tif_cfg = cfg.inference.get("output_unc_tif", None)
        if pred_tif_cfg is None:
            pred_tif_cfg = str(out_path.with_name(out_path.stem + "_predictions_20traits.tif"))
        if unc_tif_cfg is None:
            unc_tif_cfg = str(out_path.with_name(out_path.stem + "_uncertainty_20traits.tif"))
        pred_tif_path = _write_multiband_tif(
            Path(str(pred_tif_cfg)),
            pred_cube,
            transform=bundle.transform,
            crs=bundle.crs,
            logger=logger,
        )
        unc_tif_path = _write_multiband_tif(
            Path(str(unc_tif_cfg)),
            unc_cube,
            transform=bundle.transform,
            crs=bundle.crs,
            logger=logger,
        )
        out_df["pred_tif_path"] = str(pred_tif_path)
        out_df["unc_tif_path"] = str(unc_tif_path)

    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved scene prediction+uncertainty: {out_path} | rows={len(out_df)}")
    return out_path


@hydra.main(config_path="../../../configs/experiments", config_name="scene_infer_uncertainty", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_inference(cfg)


if __name__ == "__main__":
    main()
