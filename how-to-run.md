# Commands

## 1) Environment Setup

```bash
make setup
```

## 2) Download Raw Data

```bash
make data
```

## 3) Build The Processed Dataset And Features

```bash
make pipeline
```

This regenerates:

- `data/processed/train_features.csv.gz`
- `data/processed/val_features.csv.gz`
- `data/processed/test_features.csv.gz`
- `data/processed/feature_engineering_summary.json`

## 4) Run Baseline Model Experiments

```bash
uv run python scripts/run_training.py --model-name logistic_regression --experiment-name 01-baselines --run-name baseline-logistic-regression
uv run python scripts/run_training.py --model-name perceptron --experiment-name 01-baselines --run-name baseline-perceptron
uv run python scripts/run_training.py --model-name linear_regression_classifier --experiment-name 01-baselines --run-name baseline-linear-regression-classifier
```

## 5) Run Advanced Model Experiments

```bash
uv run python scripts/run_training.py --model-name random_forest --experiment-name 02-advanced-models --run-name advanced-random-forest
uv run python scripts/run_training.py --model-name svm_linear --experiment-name 02-advanced-models --run-name advanced-svm-linear
uv run python scripts/run_training.py --model-name svm_rbf --experiment-name 02-advanced-models --run-name advanced-svm-rbf
uv run python scripts/run_training.py --model-name xgboost --experiment-name 02-advanced-models --run-name advanced-xgboost
uv run python scripts/run_training.py --model-name lightgbm --experiment-name 02-advanced-models --run-name advanced-lightgbm
```


## 6) Run Optuna Hyperparameter Tuning


```bash
make tune
```

Or tune selected models only:

```bash
uv run python scripts/run_optimization.py --models random_forest,svm_rbf,xgboost,lightgbm --best-model-experiment 02-advanced-models
```

## 7) Run Leakage Ablation Audit

```bash
make ablation-audit
make ablation-audit-strict
```

## 8) MLFLOW UI

```bash
make mlflow-ui
```

Open MLflow at `http://127.0.0.1:5000`.
