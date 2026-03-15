"""Data I/O helpers: load CSV dataset and resolve column names."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def load_dataset(cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], pd.DataFrame]:
    """Load the dataset CSV and return arrays + column names.

    Returns
    -------
    spectra     : (N, 1721) float32
    targets     : (N, 20)   float32
    spectra_cols: list of 1721 column names
    trait_cols  : list of 20 trait column names
    df          : full DataFrame (needed for TransferSplitter)
    """
    path = Path(cfg.data.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. "
                                "Set data.data_path in your config.")

    df = pd.read_csv(path, low_memory=False)

    n_bands = int(cfg.data.n_spectra_bands)  # 1721
    n_traits = int(cfg.data.n_traits)        # 20

    all_cols = df.columns.tolist()

    # Spectra columns: explicit wavelength range (preferred), else positional fallback.
    if bool(cfg.data.get("use_wavelength_range", True)):
        wl_start = int(cfg.data.get("spectra_wl_start", 400))
        wl_end = int(cfg.data.get("spectra_wl_end", 2450))
        spectra_cols = [str(i) for i in range(wl_start, wl_end + 1) if str(i) in df.columns]
    else:
        start = int(cfg.data.spectra_cols_start)
        spectra_cols = all_cols[start: start + n_bands]

    # Trait columns: configured names or auto-detected as remaining columns
    if cfg.data.trait_names is not None:
        trait_cols = list(cfg.data.trait_names)
    else:
        non_spectra = [c for c in all_cols if c not in spectra_cols]
        # Drop domain / metadata columns that are not traits
        skip = {"dataset", "site", "id", "index"}
        trait_cols = [c for c in non_spectra if c.lower() not in skip][:n_traits]

    assert len(spectra_cols) == n_bands, (
        f"Expected {n_bands} spectra cols, got {len(spectra_cols)}. "
        "Check data.spectra_wl_start/spectra_wl_end or spectra_cols_start."
    )
    assert len(trait_cols) == n_traits, (
        f"Expected {n_traits} trait cols, got {len(trait_cols)}. "
        "Set data.trait_names explicitly."
    )

    spectra = df[spectra_cols].values.astype(np.float32)
    targets = df[trait_cols].values.astype(np.float32)

    return spectra, targets, spectra_cols, trait_cols, df
