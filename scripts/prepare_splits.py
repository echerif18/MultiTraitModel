"""scripts/prepare_splits.py

Pre-generate and save CV fold indices so all SLURM array tasks
use identical, reproducible splits.

Usage
-----
    python scripts/prepare_splits.py \
        --data_path data/processed/dataset.csv \
        --splits_dir data/splits \
        --n_folds 5 \
        --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate CV split indices.")
    parser.add_argument("--data_path", required=True, help="Path to dataset CSV")
    parser.add_argument("--splits_dir", default="data/splits", help="Output directory")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["kfold", "stratified"],
        default="kfold",
        help="CV split strategy.",
    )
    parser.add_argument(
        "--stratify_col",
        default="dataset",
        help="Column used for stratification when --strategy stratified.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    n = len(df)
    out = Path(args.splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = np.arange(n)
    if args.strategy == "stratified":
        labels = df[args.stratify_col].astype(str).values
        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(idx, labels)
    else:
        splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(idx)

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        np.save(out / f"fold_{fold}_train.npy", train_idx)
        np.save(out / f"fold_{fold}_val.npy", val_idx)
        print(f"Fold {fold}: train={len(train_idx):,}  val={len(val_idx):,}")

    print(f"\nSaved {args.n_folds} folds to '{out}'")


if __name__ == "__main__":
    main()
