from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from ..utils import get_logger
from ..utils.constants import (
    DEFAULT_RANDOM_STATE,
    MLRUNS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)
from ..utils.leakage import split_leaky_feature_columns
from .baseline import build_baseline_model
from .svm_model import build_svm_model
from .tree_models import build_tree_model

ALLOWED_EXPERIMENT_NAMES = (
    "01-baselines",
    "02-advanced-models",
    "05-final-champion",
)

ADVANCED_MODEL_NAMES = {
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm_linear",
    "svm_rbf",
}


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _resolve_sqlite_uri(uri: str | None, *, base_dir: Path) -> str | None:
    if not uri:
        return uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db_path = uri[len("sqlite:///") :]
        return f"sqlite:///{_resolve_path(db_path, base_dir=base_dir)}"
    return uri


def _parse_id_columns(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        cols = tuple(part.strip() for part in value.split(",") if part.strip())
        return cols or ("matchid", "teamid")

    if isinstance(value, (list, tuple)):
        parsed = tuple(str(part).strip() for part in value if str(part).strip())
        return parsed or ("matchid", "teamid")

    return ("matchid", "teamid")


@dataclass(slots=True)
class ModelConfig:
    """Model-selection configuration resolved from config files."""

    model_name: str = "logistic_regression"
    model_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        model_name_override: str | None = None,
    ) -> ModelConfig:
        """Build model config using prefixed hyperparameters.

        Hyperparameter extraction convention:
        - selected model key is in "default_model_name" (or override)
        - params are read from keys that start with "{model_name}_"
        """

        selected_model = model_name_override or str(
            mapping.get("default_model_name", "logistic_regression")
        )
        selected_model = selected_model.strip().lower()

        prefix = f"{selected_model}_"
        params: dict[str, Any] = {}
        for key, value in mapping.items():
            if key.startswith(prefix):
                params[key[len(prefix) :]] = value

        return cls(model_name=selected_model, model_params=params)


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for deterministic training orchestration and artifacts."""

    experiment_name: str = "01-baselines"
    run_name: str = "logistic-regression-baseline"
    processed_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    mlruns_dir: Path = MLRUNS_DIR
    tracking_uri: str | None = None
    target_column: str = TARGET_COLUMN
    id_columns: tuple[str, ...] = ("matchid", "teamid")
    random_state: int = DEFAULT_RANDOM_STATE

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: Path,
        experiment_name_override: str | None = None,
        run_name_override: str | None = None,
    ) -> TrainingConfig:
        """Build training config from a flat YAML mapping."""

        experiment_name = experiment_name_override or str(
            mapping.get("experiment_name", "01-baselines")
        )
        if experiment_name not in ALLOWED_EXPERIMENT_NAMES:
            allowed = ", ".join(ALLOWED_EXPERIMENT_NAMES)
            raise ValueError(
                f"Unsupported experiment_name '{experiment_name}'. "
                f"Use one of: {allowed}"
            )

        run_name = run_name_override or str(
            mapping.get("run_name", "logistic-regression-baseline")
        )

        return cls(
            experiment_name=experiment_name,
            run_name=run_name,
            processed_dir=_resolve_path(
                mapping.get("processed_dir", PROCESSED_DATA_DIR),
                base_dir=base_dir,
            ),
            models_dir=_resolve_path(
                mapping.get("models_dir", MODELS_DIR),
                base_dir=base_dir,
            ),
            mlruns_dir=_resolve_path(
                mapping.get("mlruns_dir", MLRUNS_DIR),
                base_dir=base_dir,
            ),
            tracking_uri=_resolve_sqlite_uri(
                str(mapping.get("tracking_uri", "")).strip() or None,
                base_dir=base_dir,
            ),
            target_column=str(mapping.get("target_column", TARGET_COLUMN)),
            id_columns=_parse_id_columns(mapping.get("id_columns", "matchid,teamid")),
            random_state=int(mapping.get("random_state", DEFAULT_RANDOM_STATE)),
        )


@dataclass(slots=True)
class TrainingRunResult:
    """Result payload for a complete training run."""

    experiment_name: str
    run_name: str
    run_id: str
    model_name: str
    model_path: Path
    metrics: dict[str, float]
    feature_columns: list[str]


class ModelFactory:
    """Config-driven factory that resolves model names to implementations."""

    def __init__(self, *, random_state: int = DEFAULT_RANDOM_STATE) -> None:
        self.random_state = random_state

    def create(self, model_name: str, model_params: Mapping[str, Any]) -> Any:
        """Build the selected model from configured family builders."""

        normalized = model_name.strip().lower()
        for builder in (build_baseline_model, build_tree_model, build_svm_model):
            model = builder(
                normalized,
                params=model_params,
                random_state=self.random_state,
            )
            if model is not None:
                return model

        raise ValueError(f"Unsupported model name '{normalized}'.")


def load_feature_splits(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test feature files from the processed directory."""

    train_path = processed_dir / "train_features.csv.gz"
    val_path = processed_dir / "val_features.csv.gz"
    test_path = processed_dir / "test_features.csv.gz"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected processed feature split at {path}. "
                "Run scripts/run_pipeline.py first."
            )

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def _extract_xy(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    required = [*id_columns, target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Feature split is missing required columns: " + ", ".join(sorted(missing))
        )

    candidate_feature_columns = [
        column for column in frame.columns if column not in set(required)
    ]
    feature_columns, _ = split_leaky_feature_columns(
        candidate_feature_columns,
        target_col=target_column,
    )
    if not feature_columns:
        raise ValueError("No feature columns were found in processed split.")

    x = frame[feature_columns].copy()
    y = pd.to_numeric(frame[target_column], errors="coerce")
    if y.isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains non-numeric values."
        )

    return x, y.astype(int), feature_columns


def _predict_scores(model: Any, x: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities

    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))

    return None


def _fit_with_optional_validation(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    """Fit models while allowing wrappers to consume validation data."""

    try:
        # NOTE: Advanced wrappers can use validation splits for early stopping.
        model.fit(
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    except TypeError:
        try:
            model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
        except TypeError:
            model.fit(x_train, y_train)


def _compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, y_score))

    return metrics


def _extract_feature_importances(
    model: Any,
    *,
    feature_columns: list[str],
) -> pd.DataFrame | None:
    """Build a normalized feature-importance table when the model supports it."""

    raw_importances: Any | None = None
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_

    if raw_importances is None:
        return None

    importances = np.asarray(raw_importances, dtype=float).reshape(-1)
    if importances.shape[0] != len(feature_columns):
        return None

    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )
    return frame.sort_values("importance", ascending=False, ignore_index=True)


class Trainer:
    """Train, evaluate, persist, and log models for Phase 4 workflows.

    The class enforces split discipline to keep generalization estimates clean:
    - train metrics approximate E_in
    - val/test metrics proxy E_out with strict holdout separation
    """

    def __init__(
        self,
        *,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        model_factory: ModelFactory | None = None,
    ) -> None:
        """Initialize trainer dependencies and deterministic model factory."""

        self.training_config = training_config
        self.model_config = model_config
        self.model_factory = model_factory or ModelFactory(
            random_state=self.training_config.random_state
        )
        self.logger = get_logger(__name__)
        self.model: Any | None = None

    def _persist_feature_importance(
        self,
        output_dir: Path,
        *,
        feature_columns: list[str],
    ) -> Path | None:
        """Persist feature-importance artifact for tree-style models."""

        if self.model is None:
            return None

        importance_frame = _extract_feature_importances(
            self.model,
            feature_columns=feature_columns,
        )
        if importance_frame is None:
            return None

        output_path = output_dir / "feature_importance.csv"
        importance_frame.to_csv(output_path, index=False)
        return output_path

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        show_progress: bool = False,
    ) -> dict[str, float]:
        """Fit model on train split and evaluate on train/val splits."""

        self.model = self.model_factory.create(
            self.model_config.model_name,
            self.model_config.model_params,
        )

        # NOTE: Fit is isolated to train split to avoid validation/test leakage.
        started_at = time.perf_counter()
        _fit_with_optional_validation(
            self.model,
            x_train,
            y_train,
            x_val,
            y_val,
            show_progress=(
                show_progress and self.model_config.model_name in ADVANCED_MODEL_NAMES
            ),
            progress_desc=f"{self.model_config.model_name} fit",
        )
        fit_seconds = time.perf_counter() - started_at

        train_pred = self.model.predict(x_train)
        val_pred = self.model.predict(x_val)

        train_score = _predict_scores(self.model, x_train)
        val_score = _predict_scores(self.model, x_val)

        metrics = {
            "fit_seconds": float(fit_seconds),
            **_compute_classification_metrics(
                y_train,
                train_pred,
                y_score=train_score,
                prefix="train",
            ),
            **_compute_classification_metrics(
                y_val,
                val_pred,
                y_score=val_score,
                prefix="val",
            ),
        }

        return metrics

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate the fitted model on the held-out test split only once."""

        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before evaluate().")

        test_pred = self.model.predict(x_test)
        test_score = _predict_scores(self.model, x_test)
        return _compute_classification_metrics(
            y_test,
            test_pred,
            y_score=test_score,
            prefix="test",
        )

    def persist(self, run_id: str, *, feature_columns: list[str]) -> Path:
        """Persist the fitted model and companion metadata to the models directory."""

        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before persist().")

        output_dir = (
            self.training_config.models_dir
            / self.training_config.experiment_name
            / self.model_config.model_name
            / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(self.model, handle)

        metadata_path = output_dir / "model_metadata.json"
        metadata = {
            "model_name": self.model_config.model_name,
            "experiment_name": self.training_config.experiment_name,
            "run_name": self.training_config.run_name,
            "target_column": self.training_config.target_column,
            "id_columns": list(self.training_config.id_columns),
            "feature_columns": feature_columns,
            "random_state": self.training_config.random_state,
            "lfd": {
                # NOTE: Hypothesis class and complexity are explicit for reproducible comparisons.
                "hypothesis_class_selection": (
                    "config-driven model family selection with explicit model parameters"
                ),
                # NOTE: Train/val/test metrics are separated to keep
                # NOTE: E_in and E_out diagnostics honest.
                "generalization_protocol": (
                    "train metrics represent E_in; val/test metrics represent E_out proxies"
                ),
            },
        }

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        return model_path

    def train(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        show_progress: bool = False,
    ) -> TrainingRunResult:
        """Run end-to-end fit/evaluate/persist/log pipeline with MLflow."""
        progress_desc = f"Training {self.model_config.model_name}"
        with tqdm(
            total=6,
            desc=progress_desc,
            unit="step",
            disable=not show_progress,
        ) as progress:
            x_train, y_train, feature_columns = _extract_xy(
                train_frame,
                target_column=self.training_config.target_column,
                id_columns=self.training_config.id_columns,
            )
            x_val, y_val, _ = _extract_xy(
                val_frame,
                target_column=self.training_config.target_column,
                id_columns=self.training_config.id_columns,
            )
            x_test, y_test, _ = _extract_xy(
                test_frame,
                target_column=self.training_config.target_column,
                id_columns=self.training_config.id_columns,
            )
            progress.update(1)

            tracking_uri = self.training_config.tracking_uri
            if tracking_uri is None:
                self.training_config.mlruns_dir.mkdir(parents=True, exist_ok=True)
                tracking_uri = f"file://{self.training_config.mlruns_dir.resolve()}"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.training_config.experiment_name)
            progress.update(1)

            self.logger.info(
                "Starting training run '%s' in experiment '%s'",
                self.training_config.run_name,
                self.training_config.experiment_name,
            )

            with mlflow.start_run(run_name=self.training_config.run_name) as run:
                mlflow.set_tags(
                    {
                        "model_name": self.model_config.model_name,
                        "project_scope": "post-game-team-level",
                        "lfd_generalization": "train-vs-val-vs-test",
                        "lfd_complexity_control": "config-driven-hypothesis-class",
                    }
                )

                mlflow.log_params(
                    {
                        "model_name": self.model_config.model_name,
                        "random_state": self.training_config.random_state,
                        "target_column": self.training_config.target_column,
                        "id_columns": ",".join(self.training_config.id_columns),
                        "feature_count": len(feature_columns),
                    }
                )
                if self.model_config.model_params:
                    mlflow.log_params(
                        {
                            f"model_param_{key}": value
                            for key, value in self.model_config.model_params.items()
                        }
                    )
                progress.update(1)

                fit_metrics = self.fit(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    show_progress=show_progress,
                )
                test_metrics = self.evaluate(x_test, y_test)
                metrics = {**fit_metrics, **test_metrics}
                mlflow.log_metrics(metrics)
                progress.update(1)

                model_path = self.persist(
                    run.info.run_id,
                    feature_columns=feature_columns,
                )
                mlflow.log_artifact(str(model_path), artifact_path="model")

                feature_importance_path = self._persist_feature_importance(
                    model_path.parent,
                    feature_columns=feature_columns,
                )
                if feature_importance_path is not None:
                    mlflow.log_artifact(
                        str(feature_importance_path), artifact_path="reports"
                    )
                progress.update(1)

                metrics_path = model_path.parent / "metrics.json"
                with metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2, sort_keys=True)
                    handle.write("\n")

                features_path = model_path.parent / "features.json"
                with features_path.open("w", encoding="utf-8") as handle:
                    json.dump({"feature_columns": feature_columns}, handle, indent=2)
                    handle.write("\n")

                mlflow.log_artifact(str(metrics_path), artifact_path="reports")
                mlflow.log_artifact(str(features_path), artifact_path="reports")

                self.logger.info(
                    "Completed training run '%s' (run_id=%s)",
                    self.training_config.run_name,
                    run.info.run_id,
                )
                progress.update(1)

                return TrainingRunResult(
                    experiment_name=self.training_config.experiment_name,
                    run_name=self.training_config.run_name,
                    run_id=run.info.run_id,
                    model_name=self.model_config.model_name,
                    model_path=model_path,
                    metrics=metrics,
                    feature_columns=feature_columns,
                )

    def train_from_processed_features(
        self,
        *,
        show_progress: bool = False,
    ) -> TrainingRunResult:
        """Load processed splits and execute one full training run."""

        train_frame, val_frame, test_frame = load_feature_splits(
            self.training_config.processed_dir
        )
        return self.train(
            train_frame,
            val_frame,
            test_frame,
            show_progress=show_progress,
        )
