"""Unit tests for model, loss, and data pipeline."""
import numpy as np
import pytest
import torch

from plant_trait_retrieval.models.efficientnet1d import EfficientNet1d
from plant_trait_retrieval.training.losses import CustomHuberLoss
from plant_trait_retrieval.data.dataset import HyperspectralDataset
from plant_trait_retrieval.data.preprocessing import (
    PowerTransformerWrapper,
    SpectraStandardScaler,
    build_transforms,
)


# ── Model ──────────────────────────────────────────────────────────────────

class TestEfficientNet1d:
    def test_forward_shape(self):
        model = EfficientNet1d(in_channels=1, input_length=1721, n_outputs=20)
        x = torch.randn(4, 1, 1721)
        out = model(x)
        assert out.shape == (4, 20), f"Expected (4,20), got {out.shape}"

    def test_parameter_count(self):
        model = EfficientNet1d(in_channels=1, input_length=1721, n_outputs=20)
        n = model.count_parameters()
        assert n > 0
        print(f"Parameters: {n:,}")

    def test_gradient_flow(self):
        model = EfficientNet1d(in_channels=1, input_length=1721, n_outputs=20)
        x = torch.randn(2, 1, 1721)
        y = torch.randn(2, 20)
        loss = CustomHuberLoss()(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan_output(self):
        model = EfficientNet1d()
        x = torch.randn(8, 1, 1721)
        out = model(x)
        assert not torch.isnan(out).any()


# ── Loss ───────────────────────────────────────────────────────────────────

class TestCustomHuberLoss:
    def test_zero_loss_perfect_pred(self):
        loss_fn = CustomHuberLoss(delta=1.0)
        y = torch.randn(16, 20)
        loss = loss_fn(y, y)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss(self):
        loss_fn = CustomHuberLoss(delta=1.0)
        pred = torch.randn(16, 20)
        target = torch.randn(16, 20)
        assert loss_fn(pred, target).item() > 0

    def test_delta_effect(self):
        pred = torch.ones(8, 20) * 5.0
        target = torch.zeros(8, 20)
        loss_small = CustomHuberLoss(delta=0.1)(pred, target).item()
        loss_large = CustomHuberLoss(delta=10.0)(pred, target).item()
        # smaller delta → linear regime earlier → smaller absolute loss value
        assert loss_small < loss_large


# ── Preprocessing ──────────────────────────────────────────────────────────

class TestPreprocessing:
    def _make_data(self, n=200, n_bands=1721, n_traits=20):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0, 1, (n, n_bands)).astype(np.float32)
        targets = rng.exponential(1.0, (n, n_traits)).astype(np.float32)
        return spectra, targets

    def test_spectra_scaler_zero_mean(self):
        spectra, _ = self._make_data()
        scaler = SpectraStandardScaler().fit(spectra)
        transformed = scaler.transform(spectra)
        assert np.abs(transformed.mean()) < 0.1

    def test_power_transform_roundtrip(self):
        _, targets = self._make_data()
        pt = PowerTransformerWrapper(method="yeo-johnson").fit(targets)
        transformed = pt.transform(targets)
        recovered = pt.inverse_transform(transformed)
        np.testing.assert_allclose(targets, recovered, rtol=1e-4, atol=1e-4)

    def test_build_transforms(self, tmp_path):
        spectra, targets = self._make_data()
        s_sc, t_sc = build_transforms(spectra, targets, save_dir=tmp_path)
        assert (tmp_path / "spectra_scaler.pkl").exists()
        assert (tmp_path / "target_scaler.pkl").exists()


# ── Dataset ────────────────────────────────────────────────────────────────

class TestHyperspectralDataset:
    def test_len_and_shapes(self):
        n, L, T = 50, 1721, 20
        spectra = np.random.randn(n, L).astype(np.float32)
        targets = np.random.randn(n, T).astype(np.float32)
        ds = HyperspectralDataset(spectra, targets)
        assert len(ds) == n
        x, y = ds[0]
        assert x.shape == (1, L)
        assert y.shape == (T,)
