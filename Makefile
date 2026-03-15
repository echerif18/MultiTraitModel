.PHONY: install install-dev lint format test sanity \
        splits train-cv train-transfer train-transfer-all sweep-cv-local aggregate clean

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

# ── Results ────────────────────────────────────────────────────────────────
aggregate:
	poetry run python scripts/aggregate_results.py \
		--results_dir results/cv \
		--output_dir results/figures \
		--n_folds 5

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

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist
