#!/usr/bin/env bash
set -euo pipefail

# Local CV hyperparameter sweep (non-SLURM).
#
# Usage:
#   bash scripts/sweep_cv.sh
#   MAX_RUNS=4 EPOCHS=30 bash scripts/sweep_cv.sh
#   CONFIG=cv_dataset_stratified_1721 WANDB=false bash scripts/sweep_cv.sh

CONFIG="${CONFIG:-cv_dataset_stratified_1721}"
EPOCHS="${EPOCHS:-80}"
WANDB="${WANDB:-true}"
MAX_RUNS="${MAX_RUNS:-0}" # 0 => run all

# Hyperparameter grid (edit as needed)
LR_VALUES=(1e-3 5e-4 2e-4)
WD_VALUES=(1e-4 5e-5)
DROPOUT_VALUES=(0.2 0.3)
DELTA_VALUES=(1.0 0.7)
AUG_PROB_VALUES=(0.5 0.7)
SAMPLE_WEIGHT_VALUES=(true false)

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/sweeps/cv_${TS}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/runs.log"

echo "Starting local sweep at ${TS}" | tee -a "${LOG_FILE}"
echo "Config=${CONFIG} EPOCHS=${EPOCHS} WANDB=${WANDB}" | tee -a "${LOG_FILE}"

run_idx=0
for lr in "${LR_VALUES[@]}"; do
  for wd in "${WD_VALUES[@]}"; do
    for dr in "${DROPOUT_VALUES[@]}"; do
      for delta in "${DELTA_VALUES[@]}"; do
        for augp in "${AUG_PROB_VALUES[@]}"; do
          for use_w in "${SAMPLE_WEIGHT_VALUES[@]}"; do
            run_idx=$((run_idx + 1))
            if [[ "${MAX_RUNS}" != "0" && "${run_idx}" -gt "${MAX_RUNS}" ]]; then
              echo "Reached MAX_RUNS=${MAX_RUNS}. Stopping." | tee -a "${LOG_FILE}"
              exit 0
            fi

            exp_id="sweep_${TS}_r${run_idx}_lr${lr}_wd${wd}_dr${dr}_d${delta}_a${augp}_w${use_w}"
            echo "[$(date +%H:%M:%S)] Run ${run_idx}: ${exp_id}" | tee -a "${LOG_FILE}"

            poetry run python -m plant_trait_retrieval.training.train_cv \
              --config-name "${CONFIG}" \
              training.epochs="${EPOCHS}" \
              training.learning_rate="${lr}" \
              training.weight_decay="${wd}" \
              model.dropout_rate="${dr}" \
              training.loss.delta="${delta}" \
              training.augmentation.aug_prob="${augp}" \
              training.use_sample_weights="${use_w}" \
              logging.wandb.enabled="${WANDB}" \
              +experiment_id="${exp_id}" \
              2>&1 | tee -a "${LOG_FILE}"
          done
        done
      done
    done
  done
done

echo "Sweep complete. Logs: ${LOG_FILE}" | tee -a "${LOG_FILE}"
