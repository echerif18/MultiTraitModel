from __future__ import annotations

import copy
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from ..data.loaders import make_loaders
from ..data.preprocessing import build_transforms
from ..evaluation.evaluator import predict
from ..evaluation.metrics import compute_metrics
from ..models.registry import build_model
from ..ssl import MAEDownstreamRegressor, SpectralMAE, encode_spectra, pretrain_mae
from ..training.losses import build_loss
from ..training.trainer import Trainer
from ..utils.io import load_dataset
from ..utils.misc import get_device, get_logger, seed_everything


def _maybe_load_unlabeled(cfg: DictConfig, n_bands: int) -> np.ndarray | None:
    path = cfg.experiment.get("unlabeled_data_path", None)
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    spectra_cols = [c for c in df.columns if c.isdigit()]
    if len(spectra_cols) < n_bands:
        return None
    return df[spectra_cols[:n_bands]].to_numpy(dtype=np.float32)


def _train_mae_finetune(
    cfg: DictConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> np.ndarray:
    device = str(get_device())
    mae = SpectralMAE(
        input_length=int(cfg.model.input_length),
        emb_dim=int(cfg.experiment.ssl.mae_emb_dim),
        mask_ratio=float(cfg.experiment.ssl.mask_ratio),
    )
    mae = pretrain_mae(
        mae,
        unlabeled_spectra=train_x,
        epochs=int(cfg.experiment.ssl.pretrain_epochs),
        batch_size=int(cfg.experiment.ssl.pretrain_batch_size),
        lr=float(cfg.experiment.ssl.pretrain_lr),
        device=device,
    )

    reg = MAEDownstreamRegressor(
        encoder=mae.encoder,
        emb_dim=int(cfg.experiment.ssl.mae_emb_dim),
        n_outputs=train_y.shape[1],
    ).to(device)

    x_tr = torch.from_numpy(train_x.astype(np.float32)).unsqueeze(1).to(device)
    y_tr = torch.from_numpy(train_y.astype(np.float32)).to(device)
    x_va = torch.from_numpy(val_x.astype(np.float32)).unsqueeze(1).to(device)

    opt = torch.optim.AdamW(reg.parameters(), lr=float(cfg.training.learning_rate))
    criterion = build_loss(cfg).to(device)

    reg.train()
    for _ in range(int(cfg.experiment.ssl.finetune_epochs)):
        pred = reg(x_tr)
        loss = criterion(pred, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    reg.eval()
    with torch.no_grad():
        pred_val = reg(x_va).cpu().numpy().astype(np.float32)
    return pred_val


@hydra.main(config_path="../../../configs/experiments", config_name="ssl_benchmark", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger(__name__)
    seed_everything(int(cfg.project.seed))

    spectra, targets, _, trait_cols, _ = load_dataset(cfg)
    unlabeled = _maybe_load_unlabeled(cfg, n_bands=spectra.shape[1])
    output_root = Path(cfg.experiment.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=int(cfg.data.n_folds), shuffle=True, random_state=int(cfg.project.seed))
    methods = list(cfg.experiment.methods)
    all_rows: list[pd.DataFrame] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(spectra)):
        train_x, val_x = spectra[train_idx], spectra[val_idx]
        train_y, val_y = targets[train_idx], targets[val_idx]

        spectra_scaler, target_scaler = build_transforms(
            train_spectra=train_x,
            train_targets=train_y,
            power_method=cfg.data.power_method,
            scale_spectra=bool(cfg.data.scale_spectra),
            save_dir=None,
        )
        train_xs = spectra_scaler.transform(train_x) if spectra_scaler else train_x
        val_xs = spectra_scaler.transform(val_x) if spectra_scaler else val_x

        for method in methods:
            fold_dir = output_root / method / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if method == "supervised_effnet":
                tcfg = copy.deepcopy(cfg)
                tcfg.model.name = "efficientnet1d_b0"
                tcfg.logging.wandb.enabled = False
                train_loader, val_loader = make_loaders(
                    train_spectra=train_x,
                    train_targets=train_y,
                    val_spectra=val_x,
                    val_targets=val_y,
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
                model = build_model(tcfg)
                trainer = Trainer(
                    model=model,
                    criterion=build_loss(tcfg),
                    cfg=tcfg,
                    target_transformer=target_scaler,
                    fold=fold,
                    checkpoint_dir=Path("checkpoints/ssl_benchmark/supervised"),
                    device=str(get_device()),
                )
                trainer.fit(train_loader, val_loader)
                trainer.load_best_weights()
                pred_t = predict(trainer.model, val_x, spectra_scaler, device=str(get_device()))
                pred = target_scaler.inverse_transform(pred_t)

            elif method == "mae_linear_probe":
                pretrain_pool = unlabeled if unlabeled is not None else train_xs
                mae = SpectralMAE(
                    input_length=int(cfg.model.input_length),
                    emb_dim=int(cfg.experiment.ssl.mae_emb_dim),
                    mask_ratio=float(cfg.experiment.ssl.mask_ratio),
                )
                pretrain_mae(
                    mae,
                    unlabeled_spectra=pretrain_pool,
                    epochs=int(cfg.experiment.ssl.pretrain_epochs),
                    batch_size=int(cfg.experiment.ssl.pretrain_batch_size),
                    lr=float(cfg.experiment.ssl.pretrain_lr),
                    device=str(get_device()),
                )
                z_train = encode_spectra(mae.encoder, train_xs, device=str(get_device()))
                z_val = encode_spectra(mae.encoder, val_xs, device=str(get_device()))
                pred = np.full_like(val_y, np.nan, dtype=np.float32)
                for t in range(train_y.shape[1]):
                    yt = train_y[:, t]
                    finite = np.isfinite(yt)
                    if finite.sum() < 10:
                        continue
                    ridge = Ridge(alpha=float(cfg.experiment.ssl.linear_probe_alpha))
                    ridge.fit(z_train[finite], yt[finite])
                    pred[:, t] = ridge.predict(z_val).astype(np.float32)

            elif method == "mae_finetune":
                pred = _train_mae_finetune(cfg, train_xs, train_y, val_xs, val_y)

            else:
                raise ValueError(f"Unknown SSL method: {method}")

            mdf = compute_metrics(y_true=val_y, y_pred=pred, trait_names=trait_cols)
            mdf.to_csv(fold_dir / "metrics.csv", index=False)
            mean_row = mdf[mdf["trait"] == "MEAN"].copy()
            mean_row["method"] = method
            mean_row["fold"] = fold
            all_rows.append(mean_row)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(output_root / "ssl_benchmark_long.csv", index=False)
    agg = summary.groupby("method")[["rmse", "mae", "r2", "nrmse", "bias"]].agg(["mean", "std"])
    agg.to_csv(output_root / "ssl_benchmark_summary.csv")
    logger.info("SSL benchmark complete.")


if __name__ == "__main__":
    main()
