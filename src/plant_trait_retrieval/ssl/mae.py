from __future__ import annotations

import torch
import torch.nn as nn


class SpectralMAE(nn.Module):
    """Lightweight masked autoencoder for 1D spectra."""

    def __init__(self, input_length: int = 1721, emb_dim: int = 256, mask_ratio: float = 0.4) -> None:
        super().__init__()
        self.input_length = input_length
        self.mask_ratio = mask_ratio
        self.encoder = nn.Sequential(
            nn.Linear(input_length, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, input_length),
        )

    def random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L)
        b, l = x.shape
        mask = torch.rand(b, l, device=x.device) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked, mask = self.random_mask(x)
        z = self.encoder(x_masked)
        rec = self.decoder(z)
        return rec, mask


class MAEDownstreamRegressor(nn.Module):
    """Regressor initialized from MAE encoder."""

    def __init__(self, encoder: nn.Module, emb_dim: int = 256, n_outputs: int = 20) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(emb_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = x.squeeze(1)
        z = self.encoder(x)
        return self.head(z)
