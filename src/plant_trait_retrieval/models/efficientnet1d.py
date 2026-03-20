"""1-D EfficientNet-B0 for hyperspectral regression.

Architecture faithfully re-implements the original EfficientNet-B0 block
structure adapted to 1-D convolutions, matching the TF/Keras version described
in Cherif et al. (2023) Remote Sensing of Environment.

Key design points
-----------------
* MBConv blocks with Squeeze-and-Excitation on the spectral dimension.
* Stochastic depth (drop-connect) for regularisation.
* Global average pooling → dropout → linear head (n_outputs traits).
* The input shape is (B, 1, L) where L = 1721 spectral bands.
"""
from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_connect(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth / drop-connect (applied per-sample in the batch)."""
    if not training or drop_prob == 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    # shape: (B, 1, 1) for 1-D conv feature maps
    random_tensor = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnAct1d(nn.Sequential):
    """Conv1d → BN → SiLU block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation1d(nn.Module):
    """Channel SE block adapted to 1-D."""

    def __init__(self, in_ch: int, reduced_ch: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_ch, reduced_ch, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduced_ch, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConv1d(nn.Module):
    """Mobile Inverted Bottleneck Convolution – 1-D version.

    Parameters
    ----------
    in_ch, out_ch   : input / output channels
    kernel_size     : depthwise kernel size
    stride          : depthwise stride
    expand_ratio    : expansion factor for the intermediate channels
    se_ratio        : squeeze ratio for SE (0 = disable SE)
    drop_connect    : stochastic depth probability
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (stride == 1) and (in_ch == out_ch)

        mid_ch = _make_divisible(in_ch * expand_ratio)
        se_ch = max(1, int(in_ch * se_ratio))

        layers: List[nn.Module] = []
        # Expansion phase (skip if ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBnAct1d(in_ch, mid_ch, kernel_size=1))
        # Depthwise
        layers.append(ConvBnAct1d(mid_ch, mid_ch, kernel_size=kernel_size,
                                   stride=stride, groups=mid_ch))
        # SE
        if se_ratio > 0:
            layers.append(SqueezeExcitation1d(mid_ch, se_ch))
        # Projection
        layers += [
            nn.Conv1d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = drop_connect(out, self.drop_connect_rate, self.training) + x
        return out


# ---------------------------------------------------------------------------
# EfficientNet1d
# ---------------------------------------------------------------------------

# B0 baseline block configuration (identical to the 2-D original):
#   (expand_ratio, channels, n_repeats, stride, kernel_size)
_B0_BLOCKS: List[Tuple[int, int, int, int, int]] = [
    (1,  16, 1, 1, 3),
    (6,  24, 2, 2, 3),
    (6,  40, 2, 2, 5),
    (6,  80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


class EfficientNet1d(nn.Module):
    """1-D EfficientNet.

    Parameters
    ----------
    in_channels       : number of input channels (1 for raw spectra)
    input_length      : spectral length (1721 for the dataset)
    n_outputs         : number of regression targets (20 traits)
    width_coefficient : channel scaling factor (1.0 = B0)
    depth_coefficient : depth scaling factor   (1.0 = B0)
    dropout_rate      : head dropout probability
    drop_connect_rate : total drop-connect rate (linearly scaled per block)
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_length: int = 1721,
        n_outputs: int = 20,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_outputs = n_outputs

        def round_channels(c: int) -> int:
            return _make_divisible(c * width_coefficient)

        def round_repeats(r: int) -> int:
            return int(math.ceil(depth_coefficient * r))

        # Stem
        stem_ch = round_channels(32)
        self.stem = ConvBnAct1d(in_channels, stem_ch, kernel_size=3, stride=2)

        # Count total blocks for stochastic depth scheduling
        total_blocks = sum(round_repeats(cfg[2]) for cfg in _B0_BLOCKS)
        block_idx = 0

        # Build stages
        stages: List[nn.Module] = []
        in_ch = stem_ch
        for expand_ratio, out_ch, n_repeats, stride, kernel_size in _B0_BLOCKS:
            out_ch = round_channels(out_ch)
            n_repeats = round_repeats(n_repeats)
            for i in range(n_repeats):
                dc_rate = drop_connect_rate * block_idx / total_blocks
                stages.append(
                    MBConv1d(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_connect_rate=dc_rate,
                    )
                )
                in_ch = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*stages)

        # Head
        head_ch = round_channels(1280)
        self.head_conv = ConvBnAct1d(in_ch, head_ch, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(head_ch, n_outputs)

        self._initialize_weights()

    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature embedding before dropout/regression head."""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        return self.pool(x).squeeze(-1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L)  →  (B, n_outputs)"""
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.fc(x)

    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
