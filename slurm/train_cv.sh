#!/bin/bash
#SBATCH --job-name=plant_trait_cv
#SBATCH --output=logs/slurm/cv_%A_%a.out
#SBATCH --error=logs/slurm/cv_%A_%a.err
#SBATCH --array=0-4                     # one job per fold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
# ─────────────────────────────────────────────────────────────────────────────
# Adjust --partition, --time, --mem, and --gres to your cluster's resources.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "======================================================"
echo "SLURM Job ID    : ${SLURM_JOB_ID}"
echo "SLURM Array ID  : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "SLURM Task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Node            : $(hostname)"
echo "GPU(s)          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started at      : $(date)"
echo "======================================================"

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.1         # adjust to your cluster modules
module load python/3.11

# Activate Poetry virtual environment
source "$(poetry env info --path)/bin/activate"

# Optional: set W&B API key if not already in ~/.netrc
# export WANDB_API_KEY="<your_key_here>"

FOLD=${SLURM_ARRAY_TASK_ID}

# ── Run ───────────────────────────────────────────────────────────────────────
# We override the CV splitter to run only this fold.
# The splitter loads pre-saved indices from data.cv_splits_dir,
# so all array tasks use the same deterministic splits.
python -m plant_trait_retrieval.training.train_cv \
    data.data_path="${DATA_PATH:-data/processed/dataset.csv}" \
    data.cv_splits_dir="${SPLITS_DIR:-data/splits}" \
    training.num_workers=8 \
    logging.checkpoint_dir="checkpoints/cv" \
    hydra.run.dir="outputs/cv/fold_${FOLD}" \
    +fold_override="${FOLD}"   # consumed by the script if you add single-fold support

echo "======================================================"
echo "Fold ${FOLD} finished at $(date)"
echo "======================================================"
