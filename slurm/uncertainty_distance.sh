#!/bin/bash
#SBATCH --job-name=trait_uncertainty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/uncertainty_%j.out
#SBATCH --error=logs/slurm/uncertainty_%j.err

set -euo pipefail

cd "${PROJECT_DIR:-/home/user/scratch/plant_trait_retrieval}"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

module --quiet load anaconda/3
conda activate "${CONDA_ENV:-multi-trait}"
module --quiet load cuda/12.6.0/cudnn/9.3

HYDRA_FULL_ERROR=1 poetry run python -m plant_trait_retrieval.experiments.uncertainty_eval \
  --config-name "${CONFIG_NAME:-uncertainty}" \
  data.data_path="${DATA_PATH:-data/processed/dataset.csv}" \
  experiment.base_model="${BASE_MODEL:-efficientnet1d_b0}" \
  training.epochs="${EPOCHS:-120}" \
  training.num_workers="${NUM_WORKERS:-8}"
