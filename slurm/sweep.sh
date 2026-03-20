#!/bin/bash
#SBATCH --job-name=plant_trait_sweep
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

cd "${PROJECT_DIR:-/home/mila/e/eya.cherif/scratch/plant_trait_retrieval}"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

module --quiet load anaconda/3
conda activate "${CONDA_ENV:trait}"
module --quiet load cuda/12.6.0/cudnn/9.3

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

# Grid
LR_VALUES=(1e-3 5e-4 2e-4)
WD_VALUES=(1e-4 5e-5)
DROPOUT_VALUES=(0.2 0.3)
DELTA_VALUES=(1.0 0.7)
AUG_PROB_VALUES=(0.5 0.7)
USE_W_VALUES=(true false)

TASK=${SLURM_ARRAY_TASK_ID}
N1=${#LR_VALUES[@]}
N2=${#WD_VALUES[@]}
N3=${#DROPOUT_VALUES[@]}
N4=${#DELTA_VALUES[@]}
N5=${#AUG_PROB_VALUES[@]}
N6=${#USE_W_VALUES[@]}

if [ "$TASK" -ge $((N1*N2*N3*N4*N5*N6)) ]; then
  echo "Task index ${TASK} out of range"; exit 1
fi

t=$TASK
i1=$(( t / (N2*N3*N4*N5*N6) )); t=$(( t % (N2*N3*N4*N5*N6) ))
i2=$(( t / (N3*N4*N5*N6) ));    t=$(( t % (N3*N4*N5*N6) ))
i3=$(( t / (N4*N5*N6) ));       t=$(( t % (N4*N5*N6) ))
i4=$(( t / (N5*N6) ));          t=$(( t % (N5*N6) ))
i5=$(( t / N6 ))
i6=$(( t % N6 ))

LR=${LR_VALUES[$i1]}
WD=${WD_VALUES[$i2]}
DR=${DROPOUT_VALUES[$i3]}
DELTA=${DELTA_VALUES[$i4]}
AUGP=${AUG_PROB_VALUES[$i5]}
USEW=${USE_W_VALUES[$i6]}

EXP_ID="sweep_${SLURM_ARRAY_JOB_ID}_${TASK}_lr${LR}_wd${WD}_dr${DR}_d${DELTA}_a${AUGP}_w${USEW}"
echo "Running task ${TASK}: ${EXP_ID}"

HYDRA_FULL_ERROR=1 poetry run python -m plant_trait_retrieval.training.train_cv \
  --config-name "${CONFIG_NAME:-cv_dataset_stratified_1721}" \
  data.data_path="${DATA_PATH:-data/processed/DB_50_Meta_EC_with_locations_clean1721.csv}" \
  training.learning_rate="${LR}" \
  training.weight_decay="${WD}" \
  model.dropout_rate="${DR}" \
  training.loss.delta="${DELTA}" \
  training.augmentation.aug_prob="${AUGP}" \
  training.use_sample_weights="${USEW}" \
  training.num_workers="${NUM_WORKERS:-8}" \
  training.epochs="${EPOCHS:-80}" \
  +experiment_id="${EXP_ID}"

echo "Sweep task ${TASK} complete."
