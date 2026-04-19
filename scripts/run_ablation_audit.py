from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oracle.models import ModelConfig, Trainer, TrainingConfig
from oracle.models.trainer import load_feature_splits
from oracle.utils import load_yaml_config
from oracle.utils.constants import CONFIGS_DIR

# NOTE: These are intentionally outcome-proximal objective signals for stricter leakage audit.
DEFAULT_ABLATION_PATTERNS: tuple[str, ...] = (
    "firsttower",
    "firstbaron",
    "firstinhib",
    "turretkills_sum",
    "turretkills_diff_vs_opp",
    "inhibkills_sum",
    "inhibkills_diff_vs_opp",
)

# NOTE: Profiles are ordered from strict to stricter; each later profile broadens removals.
STRICT_ABLATION_PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "objective_baseline",
        DEFAULT_ABLATION_PATTERNS,
    ),
    (
        "objective_plus_firstdragon_and_objective_diffs",
        (
            *DEFAULT_ABLATION_PATTERNS,
            "firstdragon",
            "dragonkills_diff_vs_opp",
            "baronkills_diff_vs_opp",
            "objective_control_score",
            "objective_control_rate",
        ),
    ),
    (
        "all_objective_signals",
        (
            *DEFAULT_ABLATION_PATTERNS,
            "firstdragon",
            "first",
            "objective",
            "turret",
            "tower",
            "dragon",
            "baron",
            "inhib",
            "_diff_vs_opp",
        ),
    ),
    (
        "objective_plus_macro_fight_pressure",
        (
            *DEFAULT_ABLATION_PATTERNS,
            "first",
            "objective",
            "turret",
            "tower",
            "dragon",
            "baron",
            "inhib",
            "_diff_vs_opp",
            "kills",
            "assists",
            "kda",
            "gold",
            "jungle",
            "vision",
            "timecc",
            "cc_per_min",
            "killingspree",
            "doublekills",
            "triplekills",
            "quadrakills",
            "pentakills",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-vs-ablated training audit by removing outcome-proximal objective features."
        )
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
        default="logistic_regression",
        help="Model name to audit.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="01-baselines",
        help="MLflow experiment for both runs.",
    )
    parser.add_argument(
        "--full-run-name",
        type=str,
        default="lr-audit-full",
        help="Run name for full-feature training.",
    )
    parser.add_argument(
        "--ablated-run-name",
        type=str,
        default="lr-audit-ablated-objectives",
        help="Run name for ablated-feature training.",
    )
    parser.add_argument(
        "--drop-pattern",
        action="append",
        default=None,
        help=(
            "Substring pattern to ablate. Repeatable. "
            "If omitted, defaults to objective-proximal leakage-audit patterns."
        ),
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help=(
            "Run increasingly strict ablation profiles and stop at first meaningful drop."
        ),
    )
    parser.add_argument(
        "--run-all-profiles",
        action="store_true",
        help="When --progressive is enabled, evaluate all profiles even after finding a meaningful drop.",
    )
    parser.add_argument(
        "--meaningful-drop-accuracy",
        type=float,
        default=0.03,
        help="Minimum absolute test-accuracy drop considered meaningful.",
    )
    parser.add_argument(
        "--meaningful-drop-roc-auc",
        type=float,
        default=0.005,
        help="Minimum absolute test-ROC-AUC drop considered meaningful.",
    )
    return parser.parse_args()


def _pick_columns_to_drop(
    feature_columns: list[str], patterns: tuple[str, ...]
) -> list[str]:
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    return [
        column
        for column in feature_columns
        if any(pattern in column.lower() for pattern in lowered_patterns)
    ]


def _metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_roc_auc",
        "val_roc_auc",
        "test_roc_auc",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _metric_delta(
    full_metrics: dict[str, float], ablated_metrics: dict[str, float]
) -> dict[str, float]:
    shared = sorted(set(full_metrics).intersection(ablated_metrics))
    return {
        key: float(ablated_metrics[key] - full_metrics[key])
        for key in shared
    }


def _is_meaningful_drop(
    *,
    full_metrics: dict[str, float],
    ablated_metrics: dict[str, float],
    min_accuracy_drop: float,
    min_roc_auc_drop: float,
) -> bool:
    full_acc = full_metrics.get("test_accuracy")
    ablated_acc = ablated_metrics.get("test_accuracy")
    full_auc = full_metrics.get("test_roc_auc")
    ablated_auc = ablated_metrics.get("test_roc_auc")

    accuracy_drop = (
        float(full_acc - ablated_acc)
        if full_acc is not None and ablated_acc is not None
        else 0.0
    )
    roc_auc_drop = (
        float(full_auc - ablated_auc)
        if full_auc is not None and ablated_auc is not None
        else 0.0
    )

    return accuracy_drop >= min_accuracy_drop or roc_auc_drop >= min_roc_auc_drop


def _clone_training_config(base: TrainingConfig, *, run_name: str) -> TrainingConfig:
    return TrainingConfig(
        experiment_name=base.experiment_name,
        run_name=run_name,
        processed_dir=base.processed_dir,
        models_dir=base.models_dir,
        mlruns_dir=base.mlruns_dir,
        tracking_uri=base.tracking_uri,
        target_column=base.target_column,
        id_columns=base.id_columns,
        random_state=base.random_state,
    )


def main() -> None:
    args = parse_args()

    training_mapping = load_yaml_config(args.training_config)
    model_mapping = load_yaml_config(args.model_config)

    base_training = TrainingConfig.from_mapping(
        training_mapping,
        base_dir=CONFIGS_DIR.parent,
        experiment_name_override=args.experiment_name,
    )
    model_config = ModelConfig.from_mapping(
        model_mapping,
        model_name_override=args.model_name,
    )

    train_frame, val_frame, test_frame = load_feature_splits(base_training.processed_dir)

    required_columns = set(base_training.id_columns) | {base_training.target_column}
    feature_columns = [
        column for column in train_frame.columns if column not in required_columns
    ]

    full_training = _clone_training_config(base_training, run_name=args.full_run_name)
    full_result = Trainer(training_config=full_training, model_config=model_config).train(
        train_frame, val_frame, test_frame
    )
    full_metrics = _metric_subset(full_result.metrics)

    if not args.progressive:
        patterns = (
            tuple(args.drop_pattern) if args.drop_pattern else DEFAULT_ABLATION_PATTERNS
        )
        columns_to_drop = _pick_columns_to_drop(feature_columns, patterns)

        ablated_training = _clone_training_config(
            base_training,
            run_name=args.ablated_run_name,
        )
        train_ablated = train_frame.drop(columns=columns_to_drop, errors="ignore")
        val_ablated = val_frame.drop(columns=columns_to_drop, errors="ignore")
        test_ablated = test_frame.drop(columns=columns_to_drop, errors="ignore")

        ablated_result = Trainer(
            training_config=ablated_training,
            model_config=model_config,
        ).train(train_ablated, val_ablated, test_ablated)

        ablated_metrics = _metric_subset(ablated_result.metrics)
        metric_delta = _metric_delta(full_metrics, ablated_metrics)

        audit_summary: dict[str, Any] = {
            "mode": "single",
            "experiment_name": base_training.experiment_name,
            "model_name": model_config.model_name,
            "tracking_uri": base_training.tracking_uri,
            "drop_patterns": list(patterns),
            "dropped_feature_columns": columns_to_drop,
            "full_run": {
                "run_name": full_result.run_name,
                "run_id": full_result.run_id,
                "model_path": str(full_result.model_path),
                "metrics": full_metrics,
            },
            "ablated_run": {
                "run_name": ablated_result.run_name,
                "run_id": ablated_result.run_id,
                "model_path": str(ablated_result.model_path),
                "metrics": ablated_metrics,
            },
            "delta_ablated_minus_full": metric_delta,
        }
        summary_path = base_training.processed_dir / "ablation_audit_summary.json"

    else:
        profile_summaries: list[dict[str, Any]] = []
        first_meaningful_profile: str | None = None

        for index, (profile_name, patterns) in enumerate(
            STRICT_ABLATION_PROFILES,
            start=1,
        ):
            columns_to_drop = _pick_columns_to_drop(feature_columns, patterns)
            profile_run_name = f"{args.ablated_run_name}-{index:02d}-{profile_name}"

            ablated_training = _clone_training_config(
                base_training,
                run_name=profile_run_name,
            )
            train_ablated = train_frame.drop(columns=columns_to_drop, errors="ignore")
            val_ablated = val_frame.drop(columns=columns_to_drop, errors="ignore")
            test_ablated = test_frame.drop(columns=columns_to_drop, errors="ignore")

            ablated_result = Trainer(
                training_config=ablated_training,
                model_config=model_config,
            ).train(train_ablated, val_ablated, test_ablated)

            ablated_metrics = _metric_subset(ablated_result.metrics)
            metric_delta = _metric_delta(full_metrics, ablated_metrics)
            meaningful = _is_meaningful_drop(
                full_metrics=full_metrics,
                ablated_metrics=ablated_metrics,
                min_accuracy_drop=args.meaningful_drop_accuracy,
                min_roc_auc_drop=args.meaningful_drop_roc_auc,
            )

            profile_summary = {
                "profile_name": profile_name,
                "drop_patterns": list(patterns),
                "dropped_feature_columns": columns_to_drop,
                "ablated_run": {
                    "run_name": ablated_result.run_name,
                    "run_id": ablated_result.run_id,
                    "model_path": str(ablated_result.model_path),
                    "metrics": ablated_metrics,
                },
                "delta_ablated_minus_full": metric_delta,
                "meets_meaningful_drop": meaningful,
            }
            profile_summaries.append(profile_summary)

            print(
                f"Profile {index}: {profile_name} | "
                f"dropped={len(columns_to_drop)} | "
                f"meaningful_drop={meaningful}"
            )

            if meaningful and first_meaningful_profile is None:
                first_meaningful_profile = profile_name
                if not args.run_all_profiles:
                    break

        audit_summary = {
            "mode": "progressive",
            "experiment_name": base_training.experiment_name,
            "model_name": model_config.model_name,
            "tracking_uri": base_training.tracking_uri,
            "meaningful_drop_thresholds": {
                "test_accuracy_drop": args.meaningful_drop_accuracy,
                "test_roc_auc_drop": args.meaningful_drop_roc_auc,
            },
            "full_run": {
                "run_name": full_result.run_name,
                "run_id": full_result.run_id,
                "model_path": str(full_result.model_path),
                "metrics": full_metrics,
            },
            "profiles": profile_summaries,
            "first_meaningful_profile": first_meaningful_profile,
        }
        summary_path = (
            base_training.processed_dir / "ablation_audit_progressive_summary.json"
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Ablation audit complete")
    print(f"- Summary: {summary_path}")
    print(f"- Full run: {full_result.run_id}")

    if not args.progressive:
        print(f"- Ablated run: {ablated_result.run_id}")
        print(f"- Dropped columns ({len(columns_to_drop)}):")
        for column in columns_to_drop:
            print(f"  - {column}")

        print("\nMetric comparison (ablated - full):")
        for key in sorted(metric_delta):
            print(f"  - {key}: {metric_delta[key]:+.6f}")
    else:
        print(f"- Profiles evaluated: {len(audit_summary['profiles'])}")
        print(
            "- First meaningful profile: "
            f"{audit_summary['first_meaningful_profile']}"
        )


if __name__ == "__main__":
    main()
