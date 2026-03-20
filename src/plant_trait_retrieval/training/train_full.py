from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.loaders import make_loaders
from ..data.preprocessing import build_transforms
from ..models.registry import build_model
from ..training.losses import build_loss
from ..training.trainer import Trainer
from ..utils.io import load_dataset
from ..utils.misc import get_device, get_logger, seed_everything


@hydra.main(config_path="../../../configs", config_name="full_train_1522", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(int(cfg.project.seed))
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    spectra, targets, _, trait_cols, _ = load_dataset(cfg)

    valid_target_mask = np.isfinite(targets).any(axis=1)
    valid_spectra_mask = np.isfinite(spectra).all(axis=1)
    valid_row_mask = valid_target_mask & valid_spectra_mask
    spectra = spectra[valid_row_mask]
    targets = targets[valid_row_mask]

    n = len(spectra)
    rng = np.random.default_rng(int(cfg.project.seed))
    idx = rng.permutation(n)
    n_val = max(1, int(float(cfg.training.full_train_val_fraction) * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    out_dir = Path(cfg.training.full_train_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spectra_scaler, target_scaler = build_transforms(
        train_spectra=spectra[train_idx],
        train_targets=targets[train_idx],
        power_method=cfg.data.power_method,
        scale_spectra=bool(cfg.data.scale_spectra),
        save_dir=out_dir / "scalers",
    )

    train_loader, val_loader = make_loaders(
        train_spectra=spectra[train_idx],
        train_targets=targets[train_idx],
        val_spectra=spectra[val_idx],
        val_targets=targets[val_idx],
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

    model = build_model(cfg)
    criterion = build_loss(cfg)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        cfg=cfg,
        target_transformer=target_scaler,
        fold=0,
        checkpoint_dir=out_dir / "checkpoints",
        device=str(get_device()),
    )
    history = trainer.fit(train_loader, val_loader)
    trainer.load_best_weights()

    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "trait_names": trait_cols,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        out_dir / "model_full_1522.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    logger.info(f"Saved full model artifacts to {out_dir}")


if __name__ == "__main__":
    main()
