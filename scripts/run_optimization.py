from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oracle.models import TrainingConfig
from oracle.optimization import (
    OptimizationConfig,
    get_configured_models,
    run_model_optimization,
)
from oracle.utils import load_yaml_config
from oracle.utils.constants import CONFIGS_DIR


def _parse_models(value: str | None) -> tuple[str, ...]:
    if value is None:
        return tuple()

    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    return tuple(models)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization with MLflow tracking."
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=CONFIGS_DIR / "training.yaml",
        help="Path to training config file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=CONFIGS_DIR / "model.yaml",
        help="Path to model + optimization config file.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model names override.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optional global trial budget override per model.",
    )
    parser.add_argument(
        "--best-model-experiment",
        type=str,
        default="02-advanced-models",
        help=(
            "Experiment name used by Trainer when persisting the best model artifact. "
            "Must be one of Trainer supported experiment names."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_mapping = load_yaml_config(args.training_config)
    model_mapping = load_yaml_config(args.model_config)

    training_config = TrainingConfig.from_mapping(
        training_mapping,
        base_dir=CONFIGS_DIR.parent,
        experiment_name_override=args.best_model_experiment,
        run_name_override="optuna-best-model",
    )
    optimization_config = OptimizationConfig.from_mapping(
        model_mapping,
        base_dir=CONFIGS_DIR.parent,
    )

    selected_models = _parse_models(args.models)
    models_to_optimize = selected_models or get_configured_models(model_mapping)

    successful_runs = 0
    for model_name in models_to_optimize:
        try:
            result = run_model_optimization(
                model_name=model_name,
                training_config=training_config,
                model_mapping=model_mapping,
                optimization_config=optimization_config,
                n_trials_override=args.n_trials,
            )
        except ImportError as error:
            print(f"[skip] model={model_name} dependency missing: {error}")
            continue
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"[error] model={model_name} failed: {error}")
            continue

        successful_runs += 1
        print(
            "Optimization complete: "
            f"model={result.model_name}, "
            f"study={result.study_name}, "
            f"best_value={result.best_value:.6f}, "
            f"initial_trials={result.initial_trials}, "
            f"total_trials={result.total_trials}, "
            f"added_trials={result.added_trials}, "
            f"parent_run_id={result.parent_run_id}"
        )
        print(f"  - best_params_artifact={result.best_params_path}")
        print(f"  - best_model_metadata_artifact={result.best_model_metadata_path}")
        print(f"  - study_summary_artifact={result.study_summary_path}")
        print(f"  - best_model_path={result.best_model_path}")

    if successful_runs == 0:
        raise SystemExit("No optimization studies completed successfully.")


if __name__ == "__main__":
    main()
