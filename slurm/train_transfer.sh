#!/bin/bash
#SBATCH --job-name=plant_trait_transfer
#SBATCH --output=logs/slurm/transfer_%j.out
#SBATCH --error=logs/slurm/transfer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

set -euo pipefail

echo "======================================================"
echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU(s)       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Target domain: ${TARGET_DOMAIN:-target_site}"
echo "Started at   : $(date)"
echo "======================================================"

module load cuda/12.1
module load python/3.11

source "$(poetry env info --path)/bin/activate"

# Set TARGET_DOMAIN before submitting, e.g.:
#   sbatch --export=ALL,TARGET_DOMAIN=MySite slurm/train_transfer.sh

python -m plant_trait_retrieval.training.train_transfer \
    --config-name transferability \
    data.data_path="${DATA_PATH:-data/processed/dataset.csv}" \
    data.transferability.target_domain="${TARGET_DOMAIN:-target_site}" \
    training.num_workers=8 \
    logging.checkpoint_dir="checkpoints/transfer" \
    hydra.run.dir="outputs/transfer/${TARGET_DOMAIN:-target_site}"

echo "======================================================"
echo "Transfer experiment finished at $(date)"
echo "======================================================"
