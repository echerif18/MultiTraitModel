# Plant Trait Retrieval — PyTorch Edition

**Multi-trait estimation from hyperspectral data using 1-D EfficientNet-B0.**
PyTorch rewrite of Cherif et al. (2023) *Remote Sensing of Environment*.

---

## Project structure

```
plant_trait_retrieval/
├── configs/
│   ├── base.yaml               ← default config (5-CV experiment)
│   └── transferability.yaml    ← transferability experiment config
├── src/plant_trait_retrieval/
│   ├── data/
│   │   ├── dataset.py          ← HyperspectralDataset (torch Dataset)
│   │   ├── preprocessing.py    ← PowerTransformerWrapper, SpectraStandardScaler
│   │   ├── splitter.py         ← CVSplitter, TransferSplitter
│   │   └── loaders.py          ← DataLoader factory
│   ├── models/
│   │   ├── efficientnet1d.py   ← 1-D EfficientNet-B0 (MBConv + SE)
│   │   └── registry.py         ← build_model()
│   ├── training/
│   │   ├── losses.py           ← CustomHuberLoss
│   │   ├── optim.py            ← AdamW + cosine/plateau schedulers
│   │   ├── trainer.py          ← Trainer (1-fold loop, W&B, early stopping)
│   │   ├── train_cv.py         ← 5-fold CV entry point  ← run this
│   │   └── train_transfer.py   ← transferability entry point
│   ├── evaluation/
│   │   ├── metrics.py          ← RMSE / MAE / R² / NRMSE per trait
│   │   ├── evaluator.py        ← predict() + evaluate()
│   │   └── evaluate.py         ← standalone eval CLI
│   └── utils/
│       ├── misc.py             ← seed, device, W&B helpers
│       └── io.py               ← load_dataset()
├── scripts/
│   ├── prepare_splits.py       ← pre-generate fold indices
│   ├── aggregate_results.py    ← aggregate + plot CV results
│   └── sanity_check.py         ← quick pipeline smoke test
├── slurm/
│   ├── train_cv.sh             ← SLURM array job (1 task/fold)
│   ├── train_transfer.sh       ← SLURM transferability job
│   └── sweep.sh                ← SLURM hyperparameter sweep
├── tests/
│   └── test_core.py            ← unit tests
├── Makefile
└── pyproject.toml
```

---

## 1 — Installation

### Requirements
- Python 3.10 or 3.11
- CUDA 11.8+ (for GPU training)
- [Poetry](https://python-poetry.org/docs/#installation)

### Steps

```bash
# 1. Clone your repo
git clone <your-repo-url>
cd plant_trait_retrieval

# 2. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install all dependencies
poetry install           # includes dev deps (pytest, ruff, black …)
# or for production only:
poetry install --without dev

# 4. Activate the virtual environment
poetry shell
```

---

## 2 — Data preparation

### Expected CSV format

Your dataset CSV must have:
- **1721 consecutive columns** for the hyperspectral bands (columns 0–1720 by default)
- **20 trait columns** (either the last 20 columns, or named explicitly in `configs/base.yaml`)
- An optional **`dataset`** column identifying the site/domain (required for transferability)

Example layout:

| band_0 | band_1 | … | band_1720 | LCC | LAI | … | trait_20 | dataset |
|--------|--------|---|-----------|-----|-----|---|----------|---------|

Place the file at:
```
data/processed/dataset.csv
```
Or override the path at runtime with `data.data_path=<your_path>`.

### Pre-generate CV splits (recommended before SLURM)

This ensures all SLURM array tasks share identical fold indices:

```bash
python scripts/prepare_splits.py \
    --data_path data/processed/dataset.csv \
    --splits_dir data/splits \
    --n_folds 5 \
    --seed 42
```

---

## 3 — Configuration

All settings live in `configs/base.yaml`.  
Override **any** value on the command line with Hydra dot-notation:

```bash
# Example overrides
python -m plant_trait_retrieval.training.train_cv \
    training.epochs=300 \
    training.batch_size=128 \
    training.learning_rate=5e-4 \
    training.loss.delta=0.5 \
    data.power_method=yeo-johnson
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `data.data_path` | `data/processed/dataset.csv` | Path to your CSV |
| `data.n_spectra_bands` | `1721` | Number of spectral bands |
| `data.n_traits` | `20` | Number of target traits |
| `data.power_method` | `yeo-johnson` | PowerTransformer method |
| `training.epochs` | `200` | Max training epochs |
| `training.batch_size` | `64` | Mini-batch size |
| `training.learning_rate` | `1e-3` | Initial learning rate |
| `training.loss.delta` | `1.0` | Huber loss δ |
| `training.patience` | `30` | Early stopping patience |
| `training.scheduler` | `cosine` | `cosine` / `step` / `plateau` |
| `model.dropout_rate` | `0.2` | Head dropout |
| `model.drop_connect_rate` | `0.2` | Stochastic depth rate |

---

## 4 — W&B setup

```bash
# Log in once
wandb login

# Then set your entity in configs/base.yaml:
#   logging.wandb.entity: "your-username-or-team"
# Or pass it on the CLI:
python -m plant_trait_retrieval.training.train_cv \
    logging.wandb.entity=your-entity
```

To disable W&B entirely:

```bash
python -m plant_trait_retrieval.training.train_cv \
    logging.wandb.enabled=false
```

---

## 5 — Sanity check (before long training)

```bash
python scripts/sanity_check.py
```

Expected output:
```
=== Sanity Check ===
✓  Preprocessing fitted  (spectra_scaler, target_scaler)
✓  Dataset item shapes: x=(1, 1721), y=(20,)
✓  Model instantiated  (X,XXX,XXX parameters)
✓  Forward pass  input=(8, 1, 1721) → output=(8, 20)
✓  Loss=X.XXXXXX  backward OK
✓  Inverse transform shape: (8, 20)
=== All checks passed ✓ ===
```

Or via Make:
```bash
make sanity
```

---

## 6 — Run the experiments

### 6a — 5-Fold Cross-Validation (local)

```bash
python -m plant_trait_retrieval.training.train_cv \
    data.data_path=data/processed/dataset.csv
```

Or via Make:
```bash
make train-cv DATA_PATH=data/processed/dataset.csv
```

Outputs are saved to:
```
results/cv/
├── fold_0/
│   ├── scalers/         ← fitted spectra_scaler.pkl + target_scaler.pkl
│   ├── eval/            ← fold0_val_metrics.csv, predictions.npy, targets.npy
│   └── history.csv      ← train/val loss per epoch
├── fold_1/ … fold_4/
└── all_fold_metrics.csv
```

### 6b — Transferability experiment (local)

```bash
python -m plant_trait_retrieval.training.train_transfer \
    --config-name transferability \
    data.data_path=data/processed/dataset.csv \
    data.transferability.target_domain=MyTargetSite
```

Or via Make:
```bash
make train-transfer \
    DATA_PATH=data/processed/dataset.csv \
    TARGET_DOMAIN=MyTargetSite
```

---

## 7 — Run on a SLURM cluster

### Pre-requisites on the cluster

```bash
# Upload your data
scp data/processed/dataset.csv <cluster>:~/plant_trait_retrieval/data/processed/

# Clone + install on the cluster
git clone <your-repo> && cd plant_trait_retrieval
poetry install --without dev

# Pre-generate splits
python scripts/prepare_splits.py \
    --data_path data/processed/dataset.csv \
    --splits_dir data/splits
```

### Submit 5-fold CV (one GPU job per fold, runs in parallel)

```bash
mkdir -p logs/slurm

# Option A – via Make
make slurm-cv

# Option B – direct sbatch
sbatch slurm/train_cv.sh

# With overrides via environment variable
DATA_PATH=data/processed/dataset.csv sbatch slurm/train_cv.sh
```

This submits a **SLURM array job** `#SBATCH --array=0-4`.  
Each task trains exactly one fold on its own GPU.

Monitor progress:
```bash
squeue -u $USER
tail -f logs/slurm/cv_<JOB_ID>_0.out   # fold 0 stdout
```

### Submit transferability experiment

```bash
# Set TARGET_DOMAIN before submitting
export TARGET_DOMAIN=MyTargetSite
export DATA_PATH=data/processed/dataset.csv
sbatch slurm/train_transfer.sh

# Or inline
sbatch --export=ALL,TARGET_DOMAIN=MyTargetSite,DATA_PATH=data/processed/dataset.csv \
       slurm/train_transfer.sh
```

### Submit hyperparameter sweep (12 configs × 5 folds)

```bash
sbatch slurm/sweep.sh
```

---

## 8 — Aggregate & visualise results

After all CV folds finish:

```bash
python scripts/aggregate_results.py \
    --results_dir results/cv \
    --output_dir results/figures \
    --n_folds 5
```

Or:
```bash
make aggregate
```

Produces:
```
results/
├── cv/cv_summary.csv         ← mean ± std per trait for all metrics
└── figures/
    ├── r2_per_trait.png       ← bar chart R² with error bars
    ├── rmse_per_trait.png     ← bar chart RMSE with error bars
    └── r2_heatmap.png         ← heatmap traits × folds
```

---

## 9 — Run unit tests

```bash
make test
# or
poetry run pytest tests/ -v
```

---

## 10 — Evaluate a saved checkpoint

```bash
python -m plant_trait_retrieval.evaluation.evaluate \
    checkpoint=checkpoints/cv/fold_0/fold0_epoch0150_loss0.012345.pt \
    scalers_dir=results/cv/fold_0/scalers \
    data.data_path=data/processed/dataset.csv
```

---

## 11 — Unified studies (merged repositories)

This repo now centralizes the four projects:
- `multi-traitretrieval` → baseline training/evaluation (`experiments/baseline.py`)
- `Multi_trait_Uncertainty` → distance-based uncertainty (`experiments/uncertainty_eval.py`)
- `HyspectraSSL` → SSL benchmark (MAE + downstream, `experiments/ssl_benchmark.py`)
- `multiTraitPredictions` → Gradio app workflows (`src/plant_trait_retrieval/apps/`)

Run commands:

```bash
# Study 1: baseline (EfficientNet-B0 + Single CNN + PLSR, sparse vs filled, CV tuning)
poetry run python -m plant_trait_retrieval.experiments.baseline --config-name baseline data.data_path=data/processed/dataset.csv

# Study 2: distance-based uncertainty
poetry run python -m plant_trait_retrieval.experiments.uncertainty_eval --config-name uncertainty data.data_path=data/processed/dataset.csv

# Study 3/4: SSL comparison
poetry run python -m plant_trait_retrieval.experiments.ssl_benchmark --config-name ssl_benchmark data.data_path=data/processed/dataset.csv

# EDA / quality filtering
poetry run python scripts/eda_quality_check.py --data_path data/processed/dataset.csv --out_dir results/eda

# Gradio apps
poetry run python -m plant_trait_retrieval.apps.gradio_baseline_uncertainty
poetry run python -m plant_trait_retrieval.apps.gradio_model_zoo
```

SLURM scripts for the merged studies:
- `slurm/baseline_tuning.sh`
- `slurm/uncertainty_distance.sh`
- `slurm/ssl_benchmark.sh`

---

## Architecture summary

| Component | Detail |
|---|---|
| **Backbone** | 1-D EfficientNet-B0 (7 MBConv stages, width×1.0, depth×1.0) |
| **Input** | `(B, 1, 1721)` — single-channel spectrum |
| **Output** | `(B, 20)` — 20 plant traits |
| **SE ratio** | 0.25 (per-block channel attention) |
| **Stochastic depth** | linearly scaled 0 → 0.2 across blocks |
| **Head** | GAP → Dropout(0.2) → Linear(1280, 20) |
| **Loss** | Custom Huber loss, δ=1.0, averaged over traits |
| **Target scaling** | Yeo–Johnson PowerTransformer (fit on train only) |
| **Spectra scaling** | Per-band StandardScaler (fit on train only) |
| **Optimiser** | AdamW, lr=1e-3, weight_decay=1e-4 |
| **Scheduler** | Cosine annealing with 5-epoch linear warm-up |
| **Precision** | AMP (float16) on CUDA |

---

## Citation

```bibtex
@article{cherif2023multitraitretrieval,
  title   = {From spectra to plant functional traits: Transferable multi-trait models
             from heterogeneous and sparse field data},
  author  = {Cherif, Eya and ...},
  journal = {Remote Sensing of Environment},
  year    = {2023},
  doi     = {10.1016/j.rse.2023.113612}
}
```
