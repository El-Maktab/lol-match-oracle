"""Evaluate trained models on held-out processed feature splits.

Supports both post-game and pre-game scopes using the saved processed features
and fitted preprocessors written by scripts/run_pipeline.py.
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


def _feature_path(scope: str, filename: str) -> Path:
    prefix = "" if scope == "postgame" else "pregame_"
    return PROCESSED_DATA_DIR / f"{prefix}{filename}"


def _find_postgame_model_dir() -> Path:
    """Return the preferred post-game LightGBM artifact directory."""
    pattern = str(
        Path("models") / "02-advanced-models" / "lightgbm" / "*" / "model_metadata.json"
    )
    matches = [Path(path) for path in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(
            "Could not find a post-game LightGBM model under models/02-advanced-models/lightgbm/. "
            "Run scripts/run_training.py --model-name lightgbm first or pass --model-dir explicitly."
        )

    ranked_candidates: list[tuple[bool, float, float, Path]] = []
    for meta_path in matches:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("run_name") == "optuna-best-model":
            return meta_path.parent

        metrics_path = meta_path.parent / "metrics.json"
        test_roc_auc = float("-inf")
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            test_roc_auc = float(metrics.get("test_roc_auc", float("-inf")))

        ranked_candidates.append(
            (
                meta.get("run_name") == "advanced-lightgbm",
                test_roc_auc,
                meta_path.stat().st_mtime,
                meta_path.parent,
            )
        )

    ranked_candidates.sort(reverse=True)
    return ranked_candidates[0][3]


def _find_latest_pregame_model_dir() -> Path:
    pattern = str(Path("models") / "03-pregame" / "*" / "*" / "model_metadata.json")
    matches = [Path(path) for path in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(
            "Could not find a pre-game model under models/03-pregame/. "
            "Run scripts/run_training.py --scope pregame first or pass --model-dir explicitly."
        )
    return max(matches, key=lambda path: path.stat().st_mtime).parent


def _resolve_default_model_dir(scope: str) -> Path:
    if scope == "postgame":
        return _find_postgame_model_dir()
    return _find_latest_pregame_model_dir()


def _load_model_inputs(
    *,
    model_dir: Path,
    preprocessor_path: Path,
    test_features_path: Path,
) -> tuple[dict[str, object], object, object]:
    import pandas as pd

    meta_path = model_dir / "model_metadata.json"
    with meta_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    with preprocessor_path.open("rb") as handle:
        preprocessor = pickle.load(handle)

    test_df = pd.read_csv(test_features_path)
    return metadata, preprocessor, test_df


def _align_processed_test_frame(
    test_df: object,
    *,
    feature_columns: list[str],
    target_col: str,
) -> tuple[object, object]:
    import pandas as pd

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("Expected test_df to be a pandas DataFrame.")

    aligned = test_df.copy()
    y_true = aligned[target_col].to_numpy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return y_true, aligned[feature_columns]


def _evaluate_scope(
    *,
    scope: str,
    model_dir: Path,
    preprocessor_path: Path,
    test_features_path: Path,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    metadata, preprocessor, test_df = _load_model_inputs(
        model_dir=model_dir,
        preprocessor_path=preprocessor_path,
        test_features_path=test_features_path,
    )

    feature_columns: list[str] = metadata["feature_columns"]
    target_col: str = str(metadata.get("target_column", "win"))
    if set(feature_columns).issubset(test_df.columns):
        y_true, x_test = _align_processed_test_frame(
            test_df,
            feature_columns=feature_columns,
            target_col=target_col,
        )
    else:
        y_true = test_df[target_col].to_numpy()

        preprocessor_cols: list[str] = []
        for name, _transformer, columns in preprocessor.transformers_:
            if name != "remainder":
                preprocessor_cols.extend(columns)

        for col in preprocessor_cols:
            if col not in test_df.columns:
                test_df[col] = 0.0

        x_scaled = preprocessor.transform(test_df[preprocessor_cols])
        output_column_names = [
            name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
        ]
        x_df = pd.DataFrame(x_scaled, columns=output_column_names)
        for col in feature_columns:
            if col not in x_df.columns:
                x_df[col] = 0.0
        x_test = x_df[feature_columns]

    with (model_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    y_pred = model.predict(x_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(x_test)
        y_prob = 1.0 / (1.0 + np.exp(-decision))

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    return {
        "scope": scope,
        "model_dir": model_dir,
        "model_name": metadata.get("model_name"),
        "run_name": metadata.get("run_name"),
        "feature_count": len(feature_columns),
        "metrics": metrics,
    }


def _print_comparison(results: list[dict[str, object]]) -> None:
    print("Scope        Model                Accuracy   ROC-AUC")
    print("-----------  -------------------  ---------  ---------")
    for result in results:
        metrics = result["metrics"]
        accuracy = float(metrics.get("accuracy", float("nan")))
        roc_auc = float(metrics.get("roc_auc", float("nan")))
        model_name = str(result.get("model_name") or "unknown")
        print(
            f"{str(result['scope']):<11}  {model_name:<19}  {accuracy:>9.4f}  {roc_auc:>9.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on held-out test features."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model artifact directory (contains model.pkl + model_metadata.json).",
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=("postgame", "pregame", "both"),
        default="postgame",
        help="Which evaluation scope to run.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=None,
        help="Path to the fitted ColumnTransformer pickle.",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=None,
        help="Path to the processed test feature split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scopes = ["postgame", "pregame"] if args.scope == "both" else [args.scope]
    results: list[dict[str, object]] = []

    for scope in scopes:
        model_dir = (
            args.model_dir
            if len(scopes) == 1 and args.model_dir
            else _resolve_default_model_dir(scope)
        )
        preprocessor_path = args.preprocessor or _feature_path(
            scope, "feature_preprocessor.pkl"
        )
        test_features_path = args.test_features or _feature_path(
            scope, "test_features.csv.gz"
        )
        result = _evaluate_scope(
            scope=scope,
            model_dir=model_dir,
            preprocessor_path=preprocessor_path,
            test_features_path=test_features_path,
        )
        results.append(result)

    if len(results) > 1:
        print("\nComparison")
        _print_comparison(results)
        return

    result = results[0]
    metrics = result["metrics"]
    print(f"Scope         : {result['scope']}")
    print(f"Model         : {result['model_name']} ({result['run_name']})")
    print(f"Artifact dir  : {result['model_dir']}")
    print(f"Features      : {result['feature_count']}")
    print("\nTest-set evaluation")
    for name, value in metrics.items():
        print(f"  {name:<15}: {value:.6f}")


if __name__ == "__main__":
    main()
