"""Cross-validation and transferability split utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# ---------------------------------------------------------------------------
class CVSplitter:
    """Standard K-Fold splitter with optional pre-saved indices.

    If ``splits_dir`` contains fold files (``fold_{k}_train.npy`` /
    ``fold_{k}_val.npy``) those are loaded; otherwise splits are generated
    from scratch and optionally saved.

    Parameters
    ----------
    n_folds : int
    splits_dir : Path | None
        Directory to load from / save to.
    shuffle : bool
    seed : int
    """

    def __init__(
        self,
        n_folds: int = 5,
        splits_dir: Optional[Path] = None,
        strategy: str = "kfold",
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.splits_dir = Path(splits_dir) if splits_dir else None
        self.strategy = strategy
        self.shuffle = shuffle
        self.seed = seed

    # ------------------------------------------------------------------
    def split(
        self, n_samples: int, stratify_labels: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, val_idx) for each fold."""
        if self.splits_dir and self._splits_exist():
            yield from self._load_splits()
        else:
            splits = list(self._generate_splits(n_samples, stratify_labels))
            if self.splits_dir:
                self._save_splits(splits)
            yield from splits

    # ------------------------------------------------------------------
    def _splits_exist(self) -> bool:
        return all(
            (self.splits_dir / f"fold_{k}_train.npy").exists()
            for k in range(self.n_folds)
        )

    def _load_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits = []
        for k in range(self.n_folds):
            train = np.load(self.splits_dir / f"fold_{k}_train.npy")
            val = np.load(self.splits_dir / f"fold_{k}_val.npy")
            splits.append((train, val))
        return splits

    def _save_splits(self, splits: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        for k, (train, val) in enumerate(splits):
            np.save(self.splits_dir / f"fold_{k}_train.npy", train)
            np.save(self.splits_dir / f"fold_{k}_val.npy", val)

    def _generate_splits(
        self, n_samples: int, stratify_labels: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        idx = np.arange(n_samples)
        if self.strategy == "stratified":
            if stratify_labels is None:
                raise ValueError("CVSplitter(strategy='stratified') requires stratify_labels.")
            skf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.seed,
            )
            for train_idx, val_idx in skf.split(idx, stratify_labels):
                yield train_idx, val_idx
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
            for train_idx, val_idx in kf.split(idx):
                yield train_idx, val_idx


# ---------------------------------------------------------------------------
class TransferSplitter:
    """Leave-one-site-out transferability split.

    The target domain is held out entirely as the test set; all other
    samples are used for training (or further k-fold CV).

    Parameters
    ----------
    domain_col : str
        Column in the DataFrame that identifies the domain/site.
    target_domain : str
        The domain to hold out as target.
    """

    def __init__(self, domain_col: str, target_domain: str) -> None:
        self.domain_col = domain_col
        self.target_domain = target_domain

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (source_indices, target_indices)."""
        target_mask = df[self.domain_col] == self.target_domain
        source_idx = np.where(~target_mask.values)[0]
        target_idx = np.where(target_mask.values)[0]
        return source_idx, target_idx
