#!/bin/bash
#SBATCH --job-name=plant_trait_sweep
#SBATCH --output=logs/slurm/sweep_%A_%a.out
#SBATCH --error=logs/slurm/sweep_%A_%a.err
#SBATCH --array=0-11                    # 12 configs: 3 lr × 2 bs × 2 delta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu

set -euo pipefail

module load cuda/12.1
module load python/3.11
source "$(poetry env info --path)/bin/activate"

# ── Hyperparameter grid ────────────────────────────────────────────────────
LR_VALUES=(1e-3 5e-4 1e-4)
BS_VALUES=(64 128)
DELTA_VALUES=(0.5 1.0)

# Compute combination index
TASK=${SLURM_ARRAY_TASK_ID}
N_LR=${#LR_VALUES[@]}
N_BS=${#BS_VALUES[@]}
N_DELTA=${#DELTA_VALUES[@]}

LR_IDX=$(( TASK / (N_BS * N_DELTA) ))
REMAINDER=$(( TASK % (N_BS * N_DELTA) ))
BS_IDX=$(( REMAINDER / N_DELTA ))
DELTA_IDX=$(( REMAINDER % N_DELTA ))

LR=${LR_VALUES[$LR_IDX]}
BS=${BS_VALUES[$BS_IDX]}
DELTA=${DELTA_VALUES[$DELTA_IDX]}

echo "Config ${TASK}: lr=${LR}, batch_size=${BS}, huber_delta=${DELTA}"

python -m plant_trait_retrieval.training.train_cv \
    data.data_path="${DATA_PATH:-data/processed/dataset.csv}" \
    training.learning_rate="${LR}" \
    training.batch_size="${BS}" \
    training.loss.delta="${DELTA}" \
    logging.checkpoint_dir="checkpoints/sweep/task_${TASK}" \
    hydra.run.dir="outputs/sweep/task_${TASK}"

echo "Sweep task ${TASK} done."
