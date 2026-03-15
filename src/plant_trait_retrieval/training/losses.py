"""Loss functions."""
from __future__ import annotations

import torch
import torch.nn as nn


class HuberCustomLoss(nn.Module):
    """Huber loss with finite-mask handling and optional sample weights."""

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.threshold = threshold

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ensure y_true and y_pred are on the same device
        y_true = y_true.to(y_pred.device)

        # Filter out non-finite values (infinite or NaN)
        bool_finite_out = torch.isfinite(y_pred)
        bool_finite_lb = torch.isfinite(y_true)
        finite_mask = bool_finite_out & bool_finite_lb

        if finite_mask.sum() == 0:
            # Keep gradient graph connected even when no valid targets exist in batch
            safe_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return safe_pred.sum() * 0.0

        # Residuals on valid entries
        error = y_pred[finite_mask] - y_true[finite_mask]
        abs_error = torch.abs(error)
        squared_loss = 0.5 * error**2
        linear_loss = self.threshold * abs_error - 0.5 * self.threshold**2
        is_small_error = abs_error < self.threshold
        loss = torch.where(is_small_error, squared_loss, linear_loss)

        # Optional sample weighting (per-sample, broadcast to traits)
        if sample_weight is not None and y_true.dim() == 2:
            sample_weights = torch.stack(
                [sample_weight for _ in range(y_true.size(1))], dim=1
            ).to(y_pred.device)
            weighted_loss = loss * sample_weights[finite_mask]
            denom = sample_weights[finite_mask].sum().clamp_min(1e-12)
            return weighted_loss.sum() / denom

        return loss.mean()


class LabeledLoss(nn.Module):
    """Wrapper keeping signature: (prediction, target, sample_weight=None)."""

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.huber_loss = HuberCustomLoss(threshold=threshold)

    def forward(
        self,
        lb_pred: torch.Tensor,
        y_train_lb: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.huber_loss(y_train_lb, lb_pred, sample_weight=sample_weight)


class CustomHuberLoss(HuberCustomLoss):
    """Backward-compatible alias."""
    pass


def build_loss(cfg) -> nn.Module:
    """Build loss from config."""
    name = cfg.training.loss.name.lower()
    if name == "huber":
        return LabeledLoss(threshold=float(cfg.training.loss.delta))
    elif name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss '{name}'")
