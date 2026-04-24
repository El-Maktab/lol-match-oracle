# NOTE: Just used to simplify running commands :)

.PHONY: help setup lint data pipeline pipeline-postgame pipeline-pregame train train-pregame tune ablation-audit ablation-audit-strict mlflow-ui validate evaluate evaluate-both serve

help:
	@echo "Available commands:"
	@echo ""
	@echo "  help                   : Show this help message"
	@echo "  setup                  : Create virtual environment and install dependencies"
	@echo "  data                   : Download Kaggle dataset to data/raw"
	@echo "  pipeline               : Run both post-game and pre-game pipelines"
	@echo "  pipeline-postgame      : Run only the post-game pipeline"
	@echo "  pipeline-pregame       : Run only the pre-game pipeline"
	@echo "  train                  : Run model training on processed feature splits"
	@echo "  train-pregame          : Run model training on pre-game processed feature splits"
	@echo "  tune                   : Run Optuna hyperparameter tuning with MLflow logging"
	@echo "  evaluate               : Evaluate champion model on held-out test set"
	@echo "  evaluate-both          : Compare latest/default post-game and pre-game models"
	@echo "  ablation-audit         : Run full-vs-ablated leakage audit training"
	@echo "  ablation-audit-strict  : Run progressive strict ablation profiles until meaningful drop"
	@echo "  mlflow-ui              : Launch local MLflow tracking UI"
	@echo "  validate               : Run Great Expectations data quality checkpoints"
	@echo "  lint                   : Run ruff linter on src/"
	@echo "  serve                  : Start FastAPI prediction server (uvicorn)"
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

pipeline-postgame:
	uv run python scripts/run_pipeline.py --scope postgame

pipeline-pregame:
	uv run python scripts/run_pipeline.py --scope pregame

train:
	uv run python scripts/run_training.py

train-pregame:
	uv run python scripts/run_training.py --scope pregame --experiment-name 03-pregame

tune:
	uv run python scripts/run_optimization.py

ablation-audit:
	uv run python scripts/run_ablation_audit.py

ablation-audit-strict:
	uv run python scripts/run_ablation_audit.py --progressive

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

validate:
	uv run python scripts/run_data_quality.py

evaluate:
	uv run python scripts/run_evaluation.py

evaluate-both:
	uv run python scripts/run_evaluation.py --scope both

serve:
	uv run uvicorn oracle.serving.api:app --host 0.0.0.0 --port 8000 --reload
