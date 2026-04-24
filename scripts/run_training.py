from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oracle.models import ModelConfig, Trainer, TrainingConfig
from oracle.utils import load_yaml_config
from oracle.utils.constants import CONFIGS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model training using processed feature splits."
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
        help="Path to model config file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional override for selected model key.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional MLflow experiment override.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run-name override.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=("postgame", "pregame"),
        default="postgame",
        help="Which processed feature scope to train on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_mapping = load_yaml_config(args.training_config)
    model_mapping = load_yaml_config(args.model_config)

    training_config = TrainingConfig.from_mapping(
        training_mapping,
        base_dir=CONFIGS_DIR.parent,
        experiment_name_override=args.experiment_name,
        run_name_override=args.run_name,
        scope_override=args.scope,
    )
    model_config = ModelConfig.from_mapping(
        model_mapping,
        model_name_override=args.model_name,
    )

    trainer = Trainer(
        training_config=training_config,
        model_config=model_config,
    )

    result = trainer.train_from_processed_features(show_progress=True)

    print(
        "Training complete: "
        f"experiment={result.experiment_name}, "
        f"run={result.run_name}, "
        f"run_id={result.run_id}, "
        f"model={result.model_name}, "
        f"artifact={result.model_path}"
    )
    print("Metrics:")
    for metric_name in sorted(result.metrics):
        print(f"  - {metric_name}: {result.metrics[metric_name]:.6f}")


if __name__ == "__main__":
    main()
