"""Model registry – maps config names to constructors."""
from __future__ import annotations

from omegaconf import DictConfig

from .efficientnet1d import EfficientNet1d

_REGISTRY = {
    "efficientnet1d_b0": EfficientNet1d,
}


def build_model(cfg: DictConfig) -> EfficientNet1d:
    """Instantiate a model from its Hydra config block."""
    name = cfg.model.name
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls(
        in_channels=cfg.model.in_channels,
        input_length=cfg.model.input_length,
        n_outputs=cfg.model.n_outputs,
        width_coefficient=cfg.model.width_coefficient,
        depth_coefficient=cfg.model.depth_coefficient,
        dropout_rate=cfg.model.dropout_rate,
        drop_connect_rate=cfg.model.drop_connect_rate,
    )
