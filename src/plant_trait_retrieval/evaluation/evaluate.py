"""Standalone evaluation script.

Usage
-----
    python -m plant_trait_retrieval.evaluation.evaluate \
        checkpoint=checkpoints/fold_0/best.pt \
        scalers_dir=results/cv/fold_0/scalers \
        data.data_path=data/processed/dataset.csv
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from ..data.preprocessing import PowerTransformerWrapper, SpectraStandardScaler
from ..models.registry import build_model
from ..utils.io import load_dataset
from ..utils.misc import get_device, get_logger, seed_everything
from .evaluator import evaluate


@hydra.main(config_path="../../../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(cfg.project.seed)
    device = get_device()

    checkpoint = cfg.get("checkpoint", None)
    scalers_dir = Path(cfg.get("scalers_dir", "scalers"))

    if checkpoint is None:
        raise ValueError("Provide checkpoint= path via CLI override")

    # Load data
    spectra, targets, _, trait_cols, _ = load_dataset(cfg)

    # Load scalers
    spectra_scaler = SpectraStandardScaler.load(scalers_dir / "spectra_scaler.pkl")
    target_scaler = PowerTransformerWrapper.load(scalers_dir / "target_scaler.pkl")

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint}")

    metrics_df = evaluate(
        model=model,
        spectra=spectra,
        targets=targets,
        spectra_scaler=spectra_scaler,
        target_scaler=target_scaler,
        trait_names=trait_cols,
        device=str(device),
        output_dir=Path("results/eval"),
        prefix="eval",
    )
    logger.info(f"\nMetrics:\n{metrics_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
