from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.preprocessing import SpectraStandardScaler
from ..evaluation.shap_importance import compute_shap_importance
from ..models.registry import build_model
from ..utils.io import load_dataset
from ..utils.misc import get_device, get_logger, seed_everything


def _sample_rows(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0:
        raise ValueError("Sample size must be > 0.")
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=max_n, replace=False).astype(np.int64)


@hydra.main(config_path="../../../configs/experiments", config_name="shap_eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(int(cfg.project.seed))
    device = str(get_device())

    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise ValueError("Provide checkpoint=<path>.")
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    spectra, targets, spectra_cols, trait_cols, _ = load_dataset(cfg)
    valid_target_mask = np.isfinite(targets).any(axis=1)
    valid_spectra_mask = np.isfinite(spectra).all(axis=1)
    valid_mask = valid_target_mask & valid_spectra_mask
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        spectra = spectra[valid_mask]
        targets = targets[valid_mask]
        logger.warning(
            f"Dropped unusable rows before SHAP: total_dropped={dropped} "
            f"(bad_targets={int((~valid_target_mask).sum())}, bad_spectra={int((~valid_spectra_mask).sum())})"
        )
    logger.info(f"SHAP input pool: {len(spectra)} samples | {spectra.shape[1]} bands")

    rng = np.random.default_rng(int(cfg.project.seed))
    bg_idx = _sample_rows(len(spectra), int(cfg.shap.background_size), rng)
    ev_idx = _sample_rows(len(spectra), int(cfg.shap.eval_size), rng)

    x_bg = spectra[bg_idx].astype(np.float32)
    x_ev = spectra[ev_idx].astype(np.float32)

    scalers_dir = cfg.get("scalers_dir", None)
    if scalers_dir:
        sc_path = Path(scalers_dir) / "spectra_scaler.pkl"
        if sc_path.exists():
            spectra_scaler = SpectraStandardScaler.load(sc_path)
            x_bg = spectra_scaler.transform(x_bg)
            x_ev = spectra_scaler.transform(x_ev)
            logger.info(f"Applied spectra scaler from: {sc_path}")
        else:
            logger.warning(f"scalers_dir provided but spectra scaler missing: {sc_path}. Using raw spectra.")

    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    logger.info(f"Loaded model checkpoint: {ckpt_path}")

    requested_traits = list(cfg.shap.get("trait_indices", []))
    if requested_traits:
        trait_indices = [int(i) for i in requested_traits]
    else:
        trait_indices = list(range(len(trait_cols)))

    out_dir = Path(cfg.shap.output_dir)
    wavelengths = [float(c) if str(c).isdigit() else float(i) for i, c in enumerate(spectra_cols)]
    by_trait_df, global_df = compute_shap_importance(
        model=model,
        background_spectra=x_bg,
        eval_spectra=x_ev,
        wavelengths=wavelengths,
        trait_names=trait_cols,
        trait_indices=trait_indices,
        output_dir=out_dir,
        device=device,
        n_samples=int(cfg.shap.n_samples),
    )
    logger.info(
        f"Saved SHAP outputs to {out_dir} | "
        f"traits={len(trait_indices)} | bands={len(wavelengths)} | "
        f"rows(by_trait)={len(by_trait_df)} | rows(global)={len(global_df)}"
    )


if __name__ == "__main__":
    main()
