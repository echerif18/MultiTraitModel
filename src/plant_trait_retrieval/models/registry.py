"""Model registry – maps config names to constructors."""
from __future__ import annotations

from omegaconf import DictConfig

from .efficientnet1d import EfficientNet1d
from .single_cnn1d import SingleCNN1d

_REGISTRY = {
    "efficientnet1d_b0": EfficientNet1d,
    "single_cnn1d": SingleCNN1d,
}


def build_model(cfg: DictConfig):
    """Instantiate a model from its Hydra config block."""
    name = str(cfg.model.name)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    cls = _REGISTRY[name]

    if name == "efficientnet1d_b0":
        return cls(
            in_channels=cfg.model.in_channels,
            input_length=cfg.model.input_length,
            n_outputs=cfg.model.n_outputs,
            width_coefficient=cfg.model.width_coefficient,
            depth_coefficient=cfg.model.depth_coefficient,
            dropout_rate=cfg.model.dropout_rate,
            drop_connect_rate=cfg.model.drop_connect_rate,
        )

    if name == "single_cnn1d":
        hidden = tuple(cfg.model.get("hidden_channels", [32, 64, 128]))
        return cls(
            in_channels=cfg.model.in_channels,
            input_length=cfg.model.input_length,
            n_outputs=cfg.model.n_outputs,
            hidden_channels=hidden,
            dropout_rate=cfg.model.dropout_rate,
        )

    raise RuntimeError(f"Registry entry exists but not handled for '{name}'")
