from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ..models import ModelConfig, ModelFactory, Trainer, TrainingConfig
from ..models.trainer import load_feature_splits
from ..utils.constants import DEFAULT_RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN
from ..utils.leakage import split_leaky_feature_columns
from .callbacks import (
    BestTrialMetadataCallback,
    MLflowNestedRunCallback,
    dump_study_summary,
)
from .search_spaces import get_model_trial_budget, suggest_model_params


@dataclass(slots=True)
class OptimizationConfig:
    """Configuration for reproducible Optuna + MLflow optimization runs."""

    storage_url: str = "sqlite:///optuna_studies.db"
    experiment_name: str = "03-optuna-tuning"
    study_name_prefix: str = "optuna"
    run_name_prefix: str = "optuna-study"
    metric_name: str = "val_roc_auc"
    direction: str = "maximize"
    n_trials: int = 30
    timeout_seconds: int = 0
    sampler_seed: int = DEFAULT_RANDOM_STATE
    sampler_n_startup_trials: int = 10
    pruner_n_startup_trials: int = 5
    pruner_n_warmup_steps: int = 0
    pruner_interval_steps: int = 1
    output_dir: Path = REPORTS_DIR / "optimization"
    resume_if_exists: bool = True

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> OptimizationConfig:
        """Build optimization config from flat key-value YAML mapping."""

        output_path = _resolve_path(
            mapping.get("optuna_output_dir", REPORTS_DIR / "optimization"),
            base_dir=base_dir,
        )

        return cls(
            storage_url=_resolve_sqlite_uri(
                str(mapping.get("optuna_storage_url", "sqlite:///optuna_studies.db")),
                base_dir=base_dir,
            ),
            experiment_name=str(
                mapping.get("optuna_experiment_name", "03-optuna-tuning")
            ),
            study_name_prefix=str(mapping.get("optuna_study_name_prefix", "optuna")),
            run_name_prefix=str(mapping.get("optuna_run_name_prefix", "optuna-study")),
            metric_name=str(mapping.get("optuna_metric_name", "val_roc_auc")),
            direction=str(mapping.get("optuna_direction", "maximize")),
            n_trials=max(1, int(mapping.get("optuna_n_trials", 30))),
            timeout_seconds=max(0, int(mapping.get("optuna_timeout_seconds", 0))),
            sampler_seed=int(mapping.get("optuna_sampler_seed", DEFAULT_RANDOM_STATE)),
            sampler_n_startup_trials=max(
                1, int(mapping.get("optuna_sampler_n_startup_trials", 10))
            ),
            pruner_n_startup_trials=max(
                0, int(mapping.get("optuna_pruner_n_startup_trials", 5))
            ),
            pruner_n_warmup_steps=max(
                0, int(mapping.get("optuna_pruner_n_warmup_steps", 0))
            ),
            pruner_interval_steps=max(
                1, int(mapping.get("optuna_pruner_interval_steps", 1))
            ),
            output_dir=output_path,
            resume_if_exists=bool(mapping.get("optuna_resume_if_exists", True)),
        )


@dataclass(slots=True)
class OptimizationResult:
    """Summary payload for one model optimization run."""

    model_name: str
    study_name: str
    parent_run_id: str
    best_value: float
    best_params: dict[str, Any]
    initial_trials: int
    total_trials: int
    added_trials: int
    best_model_path: Path
    best_params_path: Path
    best_model_metadata_path: Path
    study_summary_path: Path
    storage_url: str


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
        columns = tuple(part.strip() for part in value.split(",") if part.strip())
        return columns or ("matchid", "teamid")

    if isinstance(value, Sequence):
        columns = tuple(str(part).strip() for part in value if str(part).strip())
        return columns or ("matchid", "teamid")

    return ("matchid", "teamid")


def _extract_xy(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    required_columns = [*id_columns, target_column]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Processed frame is missing required columns: {missing}")

    candidate_feature_columns = [
        column for column in frame.columns if column not in set(required_columns)
    ]
    feature_columns, _ = split_leaky_feature_columns(
        candidate_feature_columns,
        target_col=target_column,
    )
    if not feature_columns:
        raise ValueError("No feature columns available in processed frame.")

    x = frame[feature_columns].copy()
    y = pd.to_numeric(frame[target_column], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Target column '{target_column}' has non-numeric values.")

    return x, y.astype(int), feature_columns


@dataclass(slots=True)
class HoldoutObjective:
    """Optuna objective that tunes on train/validation only."""

    model_name: str
    model_mapping: Mapping[str, Any]
    base_model_params: Mapping[str, Any]
    training_config: TrainingConfig
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    metric_name: str

    def __call__(self, trial: optuna.Trial) -> float:
        trial_params = suggest_model_params(
            trial,
            model_name=self.model_name,
            config_mapping=self.model_mapping,
        )
        model_params = {**self.base_model_params, **trial_params}

        trainer = Trainer(
            training_config=self.training_config,
            model_config=ModelConfig(
                model_name=self.model_name,
                model_params=model_params,
            ),
        )
        metrics = trainer.fit(
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            show_progress=False,
        )

        score = metrics.get(self.metric_name)
        if score is None:
            # NOTE: Fallback keeps optimization running if ROC-AUC is unavailable.
            score = float(metrics.get("val_f1", 0.0))

        for name, value in metrics.items():
            trial.set_user_attr(name, float(value))

        trial.report(float(score), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(score)


def _set_tracking(training_config: TrainingConfig) -> str:
    tracking_uri = training_config.tracking_uri
    if tracking_uri is None:
        training_config.mlruns_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file://{training_config.mlruns_dir.resolve()}"

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _preflight_model(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    random_state: int,
) -> None:
    factory = ModelFactory(random_state=random_state)
    factory.create(model_name, model_params)


def run_model_optimization(
    *,
    model_name: str,
    training_config: TrainingConfig,
    model_mapping: Mapping[str, Any],
    optimization_config: OptimizationConfig,
    n_trials_override: int | None = None,
) -> OptimizationResult:
    """Run one Optuna study and persist best model metadata artifacts."""

    normalized_model = model_name.strip().lower()

    base_model_params = ModelConfig.from_mapping(
        model_mapping,
        model_name_override=normalized_model,
    ).model_params

    _preflight_model(
        model_name=normalized_model,
        model_params=base_model_params,
        random_state=training_config.random_state,
    )

    train_frame, val_frame, test_frame = load_feature_splits(
        training_config.processed_dir
    )

    target_column = str(
        model_mapping.get("target_column", training_config.target_column)
    )
    if not target_column:
        target_column = TARGET_COLUMN

    id_columns = _parse_id_columns(
        model_mapping.get("id_columns", training_config.id_columns)
    )

    x_train, y_train, feature_columns = _extract_xy(
        train_frame,
        target_column=target_column,
        id_columns=id_columns,
    )
    x_val, y_val, _ = _extract_xy(
        val_frame,
        target_column=target_column,
        id_columns=id_columns,
    )
    x_test, y_test, _ = _extract_xy(
        test_frame,
        target_column=target_column,
        id_columns=id_columns,
    )

    _set_tracking(training_config)
    mlflow.set_experiment(optimization_config.experiment_name)

    study_name = f"{optimization_config.study_name_prefix}-{normalized_model}"
    sampler = TPESampler(
        seed=optimization_config.sampler_seed,
        n_startup_trials=optimization_config.sampler_n_startup_trials,
    )
    pruner = MedianPruner(
        n_startup_trials=optimization_config.pruner_n_startup_trials,
        n_warmup_steps=optimization_config.pruner_n_warmup_steps,
        interval_steps=optimization_config.pruner_interval_steps,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=optimization_config.direction,
        storage=optimization_config.storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=optimization_config.resume_if_exists,
    )

    trial_budget = (
        n_trials_override
        if n_trials_override is not None
        else get_model_trial_budget(
            normalized_model,
            config_mapping=model_mapping,
            default_n_trials=optimization_config.n_trials,
        )
    )

    initial_trials = len(study.trials)
    optimization_config.output_dir.mkdir(parents=True, exist_ok=True)

    best_snapshot_path = (
        optimization_config.output_dir / f"{study_name}_best_snapshot.json"
    )

    objective = HoldoutObjective(
        model_name=normalized_model,
        model_mapping=model_mapping,
        base_model_params=base_model_params,
        training_config=training_config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        metric_name=optimization_config.metric_name,
    )

    run_name = f"{optimization_config.run_name_prefix}-{normalized_model}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tags(
            {
                "study_name": study_name,
                "model_name": normalized_model,
                "optimization_framework": "optuna",
                "lfd_validation": "train-val-holdout",
                "lfd_data_snooping": "test-set-not-used-during-search",
            }
        )
        mlflow.log_params(
            {
                "storage_url": optimization_config.storage_url,
                "study_name": study_name,
                "direction": optimization_config.direction,
                "metric_name": optimization_config.metric_name,
                "trial_budget": int(trial_budget),
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "sampler_seed": optimization_config.sampler_seed,
                "sampler_n_startup_trials": optimization_config.sampler_n_startup_trials,
                "pruner_n_startup_trials": optimization_config.pruner_n_startup_trials,
                "pruner_n_warmup_steps": optimization_config.pruner_n_warmup_steps,
                "pruner_interval_steps": optimization_config.pruner_interval_steps,
                "initial_trials": initial_trials,
            }
        )

        study.optimize(
            objective,
            n_trials=max(1, int(trial_budget)),
            timeout=(
                None
                if optimization_config.timeout_seconds <= 0
                else optimization_config.timeout_seconds
            ),
            gc_after_trial=True,
            callbacks=[
                MLflowNestedRunCallback(
                    metric_name=optimization_config.metric_name,
                    model_name=normalized_model,
                ),
                BestTrialMetadataCallback(
                    output_path=best_snapshot_path,
                    metric_name=optimization_config.metric_name,
                ),
            ],
            show_progress_bar=False,
        )

        best_params = {**base_model_params, **study.best_params}
        best_model_config = ModelConfig(
            model_name=normalized_model,
            model_params=best_params,
        )
        best_trainer = Trainer(
            training_config=training_config,
            model_config=best_model_config,
        )
        fit_metrics = best_trainer.fit(
            x_train,
            y_train,
            x_val,
            y_val,
            show_progress=False,
        )
        test_metrics = best_trainer.evaluate(x_test, y_test)
        all_metrics = {**fit_metrics, **test_metrics}

        best_model_path = best_trainer.persist(
            parent_run.info.run_id,
            feature_columns=feature_columns,
        )

        best_params_path = (
            optimization_config.output_dir / f"{study_name}_best_params.json"
        )
        with best_params_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": normalized_model,
                    "study_name": study_name,
                    "best_trial_number": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "metric_name": optimization_config.metric_name,
                    "best_params": best_params,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        best_model_metadata_path = (
            optimization_config.output_dir / f"{study_name}_best_model_metadata.json"
        )
        with best_model_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": normalized_model,
                    "model_path": str(best_model_path),
                    "metrics": all_metrics,
                    "feature_columns": feature_columns,
                    "parent_run_id": parent_run.info.run_id,
                    "lfd": {
                        # NOTE: Best model is selected from validation signal only.
                        "model_selection": "hyperparameters-picked-from-validation-score",
                        # NOTE: Test set is evaluated once after selection to estimate E_out.
                        "generalization_estimate": "single-held-out-test-after-selection",
                    },
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        study_summary_path = dump_study_summary(
            output_path=optimization_config.output_dir
            / f"{study_name}_study_summary.json",
            study=study,
            model_name=normalized_model,
            metric_name=optimization_config.metric_name,
            initial_trials=initial_trials,
        )

        mlflow.log_metric("best_value", float(study.best_value))
        mlflow.log_metric("total_trials", float(len(study.trials)))
        mlflow.log_metric("added_trials", float(len(study.trials) - initial_trials))
        mlflow.log_metrics(
            {f"best_model_{key}": float(value) for key, value in all_metrics.items()}
        )

        mlflow.log_artifact(str(best_params_path), artifact_path="optimization")
        mlflow.log_artifact(str(best_model_metadata_path), artifact_path="optimization")
        mlflow.log_artifact(str(study_summary_path), artifact_path="optimization")
        mlflow.log_artifact(str(best_model_path), artifact_path="best_model")

        return OptimizationResult(
            model_name=normalized_model,
            study_name=study_name,
            parent_run_id=parent_run.info.run_id,
            best_value=float(study.best_value),
            best_params=best_params,
            initial_trials=initial_trials,
            total_trials=len(study.trials),
            added_trials=len(study.trials) - initial_trials,
            best_model_path=best_model_path,
            best_params_path=best_params_path,
            best_model_metadata_path=best_model_metadata_path,
            study_summary_path=study_summary_path,
            storage_url=optimization_config.storage_url,
        )


__all__ = [
    "HoldoutObjective",
    "OptimizationConfig",
    "OptimizationResult",
    "run_model_optimization",
]
