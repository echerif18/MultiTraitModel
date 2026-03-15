"""scripts/sanity_check.py

Runs a tiny end-to-end forward + backward pass with synthetic data
to verify the full pipeline is wired correctly before launching a
real training job.

Usage
-----
    python scripts/sanity_check.py
"""
from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from src.plant_trait_retrieval.data.dataset import HyperspectralDataset
from src.plant_trait_retrieval.data.preprocessing import build_transforms
from src.plant_trait_retrieval.models.efficientnet1d import EfficientNet1d
from src.plant_trait_retrieval.training.losses import CustomHuberLoss

N_TRAIN, N_VAL = 128, 32
N_BANDS, N_TRAITS = 1721, 20
BATCH = 8


def main() -> None:
    print("=== Sanity Check ===\n")

    # Synthetic data
    rng = np.random.default_rng(0)
    spectra = rng.uniform(0, 1, (N_TRAIN + N_VAL, N_BANDS)).astype(np.float32)
    targets = np.abs(rng.normal(0, 2, (N_TRAIN + N_VAL, N_TRAITS))).astype(np.float32)

    tr_spec, tr_tgt = spectra[:N_TRAIN], targets[:N_TRAIN]
    va_spec, va_tgt = spectra[N_TRAIN:], targets[N_TRAIN:]

    # Preprocessing
    s_sc, t_sc = build_transforms(tr_spec, tr_tgt)
    print(f"✓  Preprocessing fitted  (spectra_scaler, target_scaler)")

    # Dataset
    tr_X = s_sc.transform(tr_spec)
    tr_Y = t_sc.transform(tr_tgt)
    ds = HyperspectralDataset(tr_X, tr_Y)
    x0, y0 = ds[0]
    assert x0.shape == (1, N_BANDS), f"Bad spectra shape: {x0.shape}"
    assert y0.shape == (N_TRAITS,),  f"Bad target shape: {y0.shape}"
    print(f"✓  Dataset item shapes: x={tuple(x0.shape)}, y={tuple(y0.shape)}")

    # Model
    model = EfficientNet1d(in_channels=1, input_length=N_BANDS, n_outputs=N_TRAITS)
    print(f"✓  Model instantiated  ({model.count_parameters():,} parameters)")

    # Forward pass
    x_batch = torch.randn(BATCH, 1, N_BANDS)
    out = model(x_batch)
    assert out.shape == (BATCH, N_TRAITS), f"Bad output shape: {out.shape}"
    print(f"✓  Forward pass  input={tuple(x_batch.shape)} → output={tuple(out.shape)}")

    # Loss + backward
    criterion = CustomHuberLoss(delta=1.0)
    y_batch = torch.randn(BATCH, N_TRAITS)
    loss = criterion(out, y_batch)
    loss.backward()
    assert loss.item() > 0
    print(f"✓  Loss={loss.item():.6f}  backward OK")

    # PowerTransformer inverse
    arr = out.detach().numpy()
    inv = t_sc.inverse_transform(arr)
    assert inv.shape == (BATCH, N_TRAITS)
    print(f"✓  Inverse transform shape: {inv.shape}")

    print("\n=== All checks passed ✓ ===")


if __name__ == "__main__":
    main()
