#!/bin/bash
#SBATCH --job-name=plant_trait_transfer
#SBATCH --output=logs/slurm/sweep_%A_%a.out
#SBATCH --error=logs/slurm/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00

#set -euo pipefail

cd "${PROJECT_DIR:-/home/mila/e/eya.cherif/scratch/MultiTraitModel}"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

module --quiet load anaconda/3
conda activate trait
module --quiet load cuda/12.6.0/cudnn/9.3

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

# Defaults:
# - CONFIG_NAME=transferability_all_datasets_1721 runs leave-one-dataset-out transferability.
# - For single target: CONFIG_NAME=transferability and pass TARGET_DOMAIN=<name>.
HYDRA_FULL_ERROR=1 poetry run python -m plant_trait_retrieval.training.train_transfer \
  --config-name "${CONFIG_NAME:-transferability_all_datasets_1721}" \
  data.data_path="${DATA_PATH:-data/processed/DB_50_Meta_EC_with_locations_clean1721.csv}" \
  data.transferability.target_domain="${TARGET_DOMAIN:-target_site}" \
  training.num_workers="${NUM_WORKERS:-8}" \
  training.epochs="${EPOCHS:-100}" \
  +experiment_id="${EXPERIMENT_ID:-transfer_$(date +%Y%m%d_%H%M%S)}"

echo "Transfer training complete."
