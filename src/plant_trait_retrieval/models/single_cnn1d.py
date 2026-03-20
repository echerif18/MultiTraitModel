from __future__ import annotations

import torch
import torch.nn as nn


class SingleCNN1d(nn.Module):
    """Simple 1D CNN baseline for hyperspectral multi-trait regression."""

    def __init__(
        self,
        in_channels: int = 1,
        input_length: int = 1721,
        n_outputs: int = 20,
        hidden_channels: tuple[int, int, int] = (32, 64, 128),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3 = hidden_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(c3, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(start_dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
