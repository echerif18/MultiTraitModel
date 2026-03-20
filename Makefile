.PHONY: install install-dev lint format test sanity \
        splits train-cv train-transfer train-transfer-all sweep-cv-local aggregate aggregate-transfer \
        eda process-raw baseline-exp uncertainty-exp ssl-benchmark shap-eval app-baseline app-model-zoo app-disun-scene scene-infer-uncertainty \
        train-full-1522 slurm-baseline slurm-uncertainty slurm-ssl clean

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	poetry install --without dev

install-dev:
	poetry install

# ── Code quality ───────────────────────────────────────────────────────────
lint:
	poetry run ruff check src/ tests/ scripts/
	poetry run mypy src/

format:
	poetry run black src/ tests/ scripts/
	poetry run ruff check --fix src/ tests/ scripts/

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v

sanity:
	poetry run python scripts/sanity_check.py

# ── Data ───────────────────────────────────────────────────────────────────
splits:
	poetry run python scripts/prepare_splits.py \
		--data_path $(DATA_PATH) \
		--splits_dir data/splits \
		--n_folds 5 \
		--seed 42

# ── Training ───────────────────────────────────────────────────────────────
train-cv:
	poetry run python -m plant_trait_retrieval.training.train_cv \
		data.data_path=$(DATA_PATH)

train-transfer:
	poetry run python -m plant_trait_retrieval.training.train_transfer \
		data.data_path=$(DATA_PATH) \
		data.transferability.target_domain=$(TARGET_DOMAIN)

train-transfer-all:
	poetry run python -m plant_trait_retrieval.training.train_transfer \
		--config-name transferability_all_datasets_1721

sweep-cv-local:
	bash scripts/sweep_cv.sh

baseline-exp:
	poetry run python -m plant_trait_retrieval.experiments.baseline \
		--config-name baseline \
		data.data_path=$(DATA_PATH)

uncertainty-exp:
	poetry run python -m plant_trait_retrieval.experiments.uncertainty_eval \
		--config-name uncertainty \
		data.data_path=$(DATA_PATH)

ssl-benchmark:
	poetry run python -m plant_trait_retrieval.experiments.ssl_benchmark \
		--config-name ssl_benchmark \
		data.data_path=$(DATA_PATH) \
		experiment.unlabeled_data_path=$(UNLABELED_DATA_PATH)

shap-eval:
	poetry run python -m plant_trait_retrieval.experiments.shap_eval \
		--config-name shap_eval \
		checkpoint=$(CHECKPOINT) \
		scalers_dir=$(SCALERS_DIR) \
		data.data_path=$(DATA_PATH)

train-full-1522:
	poetry run python -m plant_trait_retrieval.training.train_full \
		--config-name full_train_1522 \
		data.data_path=$(DATA_PATH)

# ── Results ────────────────────────────────────────────────────────────────
aggregate:
	poetry run python scripts/aggregate_results.py \
		--results_dir results/cv \
		--output_dir results/figures \
		--n_folds 5

aggregate-transfer:
	poetry run python scripts/aggregate_transfer_results.py

# ── SLURM ──────────────────────────────────────────────────────────────────
slurm-cv:
	mkdir -p logs/slurm
	sbatch slurm/train_cv.sh

slurm-transfer:
	mkdir -p logs/slurm
	sbatch slurm/train_transfer.sh

slurm-sweep:
	mkdir -p logs/slurm
	sbatch slurm/sweep.sh

slurm-baseline:
	mkdir -p logs/slurm
	sbatch slurm/baseline_tuning.sh

slurm-uncertainty:
	mkdir -p logs/slurm
	sbatch slurm/uncertainty_distance.sh

slurm-ssl:
	mkdir -p logs/slurm
	sbatch slurm/ssl_benchmark.sh

eda:
	poetry run python scripts/eda_quality_check.py \
		--data_path $(DATA_PATH) \
		--out_dir $(OUT_DIR)

process-raw:
	poetry run python scripts/process_raw_to_processed.py

app-baseline:
	poetry run python -m plant_trait_retrieval.apps.gradio_baseline_uncertainty

app-model-zoo:
	poetry run python -m plant_trait_retrieval.apps.gradio_model_zoo

app-disun-scene:
	poetry run python -m plant_trait_retrieval.apps.gradio_disun_scene

scene-infer-uncertainty:
	poetry run python -m plant_trait_retrieval.experiments.uncertainty_scene_infer \
		--config-name scene_infer_uncertainty \
		inference.scene_csv=$(SCENE_CSV)

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist
