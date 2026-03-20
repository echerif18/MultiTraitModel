from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def encode_spectra(encoder: nn.Module, spectra: np.ndarray, device: str = "cpu") -> np.ndarray:
    encoder = encoder.to(device).eval()
    x = torch.from_numpy(spectra.astype(np.float32)).to(device)
    z = encoder(x)
    return z.cpu().numpy().astype(np.float32)


def pretrain_mae(
    model: nn.Module,
    unlabeled_spectra: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ds = TensorDataset(torch.from_numpy(unlabeled_spectra.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for (x,) in dl:
            x = x.to(device)
            rec, mask = model(x)
            # reconstruction loss only on masked positions
            loss = ((rec[mask] - x[mask]) ** 2).mean() if mask.any() else ((rec - x) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model
