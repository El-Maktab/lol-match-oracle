from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import optuna  # type: ignore[import-not-found]
from optuna.trial import FrozenTrial, TrialState  # type: ignore[import-not-found]


@dataclass(slots=True)
class MLflowNestedRunCallback:
    """Log Optuna trial results into nested MLflow child runs."""

    metric_name: str
    model_name: str

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        run_name = f"trial-{trial.number:04d}"

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tags(
                {
                    "study_name": study.study_name,
                    "model_name": self.model_name,
                    "trial_state": trial.state.name.lower(),
                    "lfd_validation_discipline": "holdout-validation-for-model-selection",
                }
            )

            if trial.params:
                safe_params = {key: str(value) for key, value in trial.params.items()}
                mlflow.log_params(safe_params)

            if trial.value is not None and trial.state == TrialState.COMPLETE:
                mlflow.log_metric(self.metric_name, float(trial.value))

            if trial.duration is not None:
                mlflow.log_metric(
                    "trial_duration_seconds", trial.duration.total_seconds()
                )

            for attr_name, attr_value in trial.user_attrs.items():
                if isinstance(attr_value, bool):
                    mlflow.log_param(f"attr_{attr_name}", str(attr_value).lower())
                    continue

                if isinstance(attr_value, (int, float)):
                    mlflow.log_metric(f"attr_{attr_name}", float(attr_value))
                else:
                    mlflow.log_param(f"attr_{attr_name}", str(attr_value))


@dataclass(slots=True)
class BestTrialMetadataCallback:
    """Persist a small JSON checkpoint whenever a new best trial is found."""

    output_path: Path
    metric_name: str

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return

        if study.best_trial.number != trial.number:
            return

        payload = {
            "study_name": study.study_name,
            "best_trial_number": study.best_trial.number,
            "best_value": float(study.best_value),
            "metric_name": self.metric_name,
            "best_params": study.best_params,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


def dump_study_summary(
    *,
    output_path: Path,
    study: optuna.Study,
    model_name: str,
    metric_name: str,
    initial_trials: int,
) -> Path:
    """Write final study summary JSON artifact."""

    completed_trials = sum(
        1 for trial in study.trials if trial.state == TrialState.COMPLETE
    )
    pruned_trials = sum(1 for trial in study.trials if trial.state == TrialState.PRUNED)
    failed_trials = sum(1 for trial in study.trials if trial.state == TrialState.FAIL)

    payload: dict[str, Any] = {
        "study_name": study.study_name,
        "model_name": model_name,
        "metric_name": metric_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "initial_trials": int(initial_trials),
        "total_trials": len(study.trials),
        "added_trials": len(study.trials) - int(initial_trials),
        "completed_trials": completed_trials,
        "pruned_trials": pruned_trials,
        "failed_trials": failed_trials,
        "lfd": {
            # NOTE: Validation-only optimization keeps test data untouched.
            "validation_discipline": "optuna-objective-uses-train-and-validation-only",
            # NOTE: Test metrics are produced only once after selecting best params.
            "data_snooping_prevention": "no-test-set-feedback-in-hyperparameter-search",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return output_path


__all__ = [
    "BestTrialMetadataCallback",
    "MLflowNestedRunCallback",
    "dump_study_summary",
]
