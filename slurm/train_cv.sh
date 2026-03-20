#!/bin/bash
#SBATCH --job-name=plant_trait_cv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/cv_%j.out
#SBATCH --error=logs/slurm/cv_%j.err

set -euo pipefail

cd "${PROJECT_DIR:-/home/user/scratch/plant_trait_retrieval}"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

module --quiet load anaconda/3
conda activate "${CONDA_ENV:-multi-trait}"
module --quiet load cuda/12.6.0/cudnn/9.3

# Optional: load .env if present
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "======================================================"
echo "SLURM Job ID : ${SLURM_JOB_ID:-N/A}"
echo "Node         : $(hostname)"
echo "Started at   : $(date)"
echo "======================================================"

HYDRA_FULL_ERROR=1 poetry run python -m plant_trait_retrieval.training.train_cv \
  --config-name "${CONFIG_NAME:-cv_dataset_stratified_1721}" \
  data.data_path="${DATA_PATH:-data/processed/DB_50_Meta_EC_with_locations_clean1721.csv}" \
  training.num_workers="${NUM_WORKERS:-8}" \
  training.epochs="${EPOCHS:-200}" \
  +experiment_id="${EXPERIMENT_ID:-cv_$(date +%Y%m%d_%H%M%S)}"

echo "CV training complete."
