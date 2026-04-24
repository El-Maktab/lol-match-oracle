"""Evaluate the champion LightGBM model on the held-out test set.

Loads the optuna-best-model artifact, applies the saved preprocessor to
test_features.csv.gz, computes classification metrics, and prints a summary.

Usage:
    make evaluate
    uv run python scripts/run_evaluation.py
    uv run python scripts/run_evaluation.py --model-dir models/02-advanced-models/lightgbm/<run-id>
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oracle.evaluation.metrics import calculate_classification_metrics
from oracle.utils.constants import PROCESSED_DATA_DIR


def _find_champion_model_dir() -> Path:
    """Return the directory of the optuna-best-model LightGBM artifact."""
    pattern = str(
        Path("models") / "02-advanced-models" / "lightgbm" / "*" / "model_metadata.json"
    )
    for meta_path in glob.glob(pattern):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("run_name") == "optuna-best-model":
            return Path(meta_path).parent
    raise FileNotFoundError(
        "Could not find optuna-best-model under models/02-advanced-models/lightgbm/. "
        "Run `make tune` first or pass --model-dir explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate champion model on the held-out test set."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model artifact directory (contains model.pkl + model_metadata.json).",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=PROCESSED_DATA_DIR / "feature_preprocessor.pkl",
        help="Path to the fitted ColumnTransformer pickle.",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=PROCESSED_DATA_DIR / "test_features.csv.gz",
        help="Path to the processed test feature split.",
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()

    # -- Resolve model directory --
    model_dir = args.model_dir or _find_champion_model_dir()

    meta_path = model_dir / "model_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    feature_columns: list[str] = metadata["feature_columns"]
    target_col: str = metadata.get("target_column", "win")

    print(f"Champion model : {metadata.get('model_name')} ({metadata.get('run_name')})")
    print(f"Artifact dir   : {model_dir}")
    print(f"Features       : {len(feature_columns)}")

    # -- Load preprocessor --
    with open(args.preprocessor, "rb") as f:
        preprocessor = pickle.load(f)

    # -- Load test split --
    test_df = pd.read_csv(args.test_features)
    y_true = test_df[target_col].to_numpy()

    # Align to preprocessor input columns
    preprocessor_cols: list[str] = []
    for name, _transformer, columns in preprocessor.transformers_:
        if name != "remainder":
            preprocessor_cols.extend(columns)

    for col in preprocessor_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    x_scaled = preprocessor.transform(test_df[preprocessor_cols])

    # Rebuild named DataFrame so we can slice to model feature columns
    out_col_names = [
        name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
    ]
    import numpy as np

    x_df = pd.DataFrame(x_scaled, columns=out_col_names)
    for col in feature_columns:
        if col not in x_df.columns:
            x_df[col] = 0.0
    x_test = x_df[feature_columns]

    # -- Load model --
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # -- Compute metrics --
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    print("\n── Test-set evaluation ──────────────────────────────────")
    for name, value in metrics.items():
        print(f"  {name:<15}: {value:.6f}")
    print("─────────────────────────────────────────────────────────")

    # NOTE: E_out estimate — this is the first and only time the test set is touched.
    print(
        f"\nROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}  |  "
        f"Accuracy: {metrics.get('accuracy', float('nan')):.4f}  |  "
        f"Brier: {metrics.get('brier_score', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
