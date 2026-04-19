# NOTE: Just used to simplify running commands :)

.PHONY: help setup lint data pipeline train ablation-audit ablation-audit-strict mlflow-ui validate

help:
	@echo "Available commands:"
	@echo ""
	@echo "  help                   : Show this help message"
	@echo "  setup                  : Create virtual environment and install dependencies"
	@echo "  data                   : Download Kaggle dataset to data/raw"
	@echo "  pipeline               : Run the data pipeline and write split datasets"
	@echo "  train                  : Run model training on processed feature splits"
	@echo "  ablation-audit         : Run full-vs-ablated leakage audit training"
	@echo "  ablation-audit-strict  : Run progressive strict ablation profiles until meaningful drop"
	@echo "  mlflow-ui              : Launch local MLflow tracking UI"
	@echo "  validate               : Run Great Expectations data quality checkpoints"
	@echo "  lint                   : Run lint" 
	@echo ""

setup:
	uv venv --python 3.12
	uv sync

lint:
	uv run ruff check src

data:
	uv run python scripts/download_data.py

pipeline:
	uv run python scripts/run_pipeline.py

train:
	uv run python scripts/run_training.py

ablation-audit:
	uv run python scripts/run_ablation_audit.py

ablation-audit-strict:
	uv run python scripts/run_ablation_audit.py --progressive

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

validate:
	uv run python scripts/run_data_quality.py