from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm.auto import tqdm

from oracle.data import (
    clean_match_dataset,
    load_raw_tables,
    merge_match_level_dataset,
    merge_pregame_dataset,
    split_train_val_test,
)
from oracle.features import build_feature_datasets, build_pregame_feature_datasets
from oracle.utils import get_logger, load_data_config
from oracle.utils.constants import (
    EXPECTED_TEAM_ROWS_PER_MATCH,
    TEAM_IDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the League of Legends data pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the data config file (defaults to configs/data.yaml).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for a faster local smoke test.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=("postgame", "pregame", "both"),
        default="both",
        help="Which pipeline scope to run.",
    )
    return parser.parse_args()


def _scope_prefix(scope: str) -> str:
    return "" if scope == "postgame" else "pregame_"


def _selected_scopes(scope: str) -> list[str]:
    return ["postgame", "pregame"] if scope == "both" else [scope]


def _run_scope_pipeline(
    scope: str,
    *,
    tables: dict[str, pd.DataFrame],
    config_path: Path | None,
    config: Any,
) -> dict[str, Any]:
    logger = get_logger(__name__)

    if scope == "postgame":
        logger.info("Merging post-game match-level dataset")
        merged = merge_match_level_dataset(
            tables["matches"],
            tables["participants"],
            tables["stats"],
            tables["teamstats"],
        )
        feature_builder = build_feature_datasets
    else:
        logger.info("Merging pre-game draft-level dataset")
        merged = merge_pregame_dataset(
            tables["matches"],
            tables["participants"],
            tables["stats"],
            tables["teambans"],
        )
        feature_builder = build_pregame_feature_datasets

    logger.info("Cleaning %s dataset", scope)
    cleaned = clean_match_dataset(
        merged,
        min_duration_seconds=config.min_curated_duration_seconds,
    )

    if config.target_column not in cleaned.columns:
        raise ValueError(
            f"Target column '{config.target_column}' not found after cleaning {scope} dataset."
        )

    logger.info("Splitting %s dataset into train/val/test sets", scope)
    splits = split_train_val_test(
        cleaned,
        target_col=config.target_column,
        group_col=config.group_column,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )

    prefix = _scope_prefix(scope)
    interim_dir = config.interim_dir
    processed_dir = config.processed_dir
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    merged_path = interim_dir / f"{prefix}match_level_dataset.csv.gz"
    train_path = processed_dir / f"{prefix}train.csv.gz"
    val_path = processed_dir / f"{prefix}val.csv.gz"
    test_path = processed_dir / f"{prefix}test.csv.gz"

    logger.info("Writing %s interim dataset to %s", scope, merged_path)
    _write_csv_gz(cleaned, merged_path)

    train_frame = pd.concat(
        [splits.x_train, splits.y_train.rename(config.target_column)], axis=1
    )
    val_frame = pd.concat(
        [splits.x_val, splits.y_val.rename(config.target_column)], axis=1
    )
    test_frame = pd.concat(
        [splits.x_test, splits.y_test.rename(config.target_column)], axis=1
    )

    logger.info("Writing %s processed splits", scope)
    _write_csv_gz(train_frame, train_path)
    _write_csv_gz(val_frame, val_path)
    _write_csv_gz(test_frame, test_path)

    logger.info("Building %s feature datasets", scope)
    feature_summary = feature_builder(
        train_frame,
        val_frame,
        test_frame,
        processed_dir=processed_dir,
        target_col=config.target_column,
        output_prefix=prefix,
    )

    return {
        "scope": scope,
        "config": {
            "config_path": str(config_path) if config_path is not None else None,
            "target_column": config.target_column,
            "group_column": config.group_column,
            "test_size": config.test_size,
            "val_size": config.val_size,
            "random_state": config.random_state,
            "min_curated_duration_seconds": config.min_curated_duration_seconds,
        },
        "merged": _frame_summary(cleaned, target_col=config.target_column),
        "splits": {
            "train": _frame_summary(train_frame, target_col=config.target_column),
            "val": _frame_summary(val_frame, target_col=config.target_column),
            "test": _frame_summary(test_frame, target_col=config.target_column),
        },
        "team_level_integrity": {
            "merged": _team_integrity_summary(cleaned, target_col=config.target_column),
            "train": _team_integrity_summary(
                train_frame, target_col=config.target_column
            ),
            "val": _team_integrity_summary(val_frame, target_col=config.target_column),
            "test": _team_integrity_summary(
                test_frame, target_col=config.target_column
            ),
        },
        "feature_engineering": {
            "summary_file": str(
                processed_dir / f"{prefix}feature_engineering_summary.json"
            ),
            "rows": feature_summary["rows"],
            "feature_count": feature_summary["feature_counts"]["final_selected"],
            "feature_counts": feature_summary["feature_counts"],
            "outputs": feature_summary["outputs"],
        },
        "output_files": {
            "merged": str(merged_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }


def _write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def _frame_summary(frame: pd.DataFrame, *, target_col: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": int(frame.isna().sum().sum()),
    }

    if target_col in frame.columns:
        target_counts = frame[target_col].value_counts(dropna=False).sort_index()
        summary["target_distribution"] = {
            str(index): int(value) for index, value in target_counts.items()
        }
        summary["target_rate"] = {
            str(index): round(float(value) / len(frame), 6) if len(frame) else 0.0
            for index, value in target_counts.items()
        }

    return summary


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _team_integrity_summary(frame: pd.DataFrame, *, target_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "team_rows_per_match_expected": EXPECTED_TEAM_ROWS_PER_MATCH,
            "duplicate_match_team_rows": 0,
            "invalid_teamid_rows": 0,
            "invalid_match_team_count": 0,
            "invalid_winner_matches": 0,
            "unique_matches": 0,
        }

    duplicate_keys = int(frame.duplicated(subset=["matchid", "teamid"]).sum())
    invalid_team_rows = int(
        (~pd.to_numeric(frame["teamid"], errors="coerce").isin(TEAM_IDS)).sum()
    )

    teams_per_match = frame.groupby("matchid")["teamid"].nunique(dropna=False)
    wins_per_match = (
        pd.to_numeric(frame[target_col], errors="coerce")
        .fillna(0)
        .groupby(frame["matchid"])
        .sum(min_count=1)
    )

    return {
        "team_rows_per_match_expected": EXPECTED_TEAM_ROWS_PER_MATCH,
        "duplicate_match_team_rows": duplicate_keys,
        "invalid_teamid_rows": invalid_team_rows,
        "invalid_match_team_count": int(
            (teams_per_match != EXPECTED_TEAM_ROWS_PER_MATCH).sum()
        ),
        "invalid_winner_matches": int((wins_per_match != 1).sum()),
        "unique_matches": int(teams_per_match.index.nunique()),
    }


def build_pipeline(
    config_path: Path | None = None,
    nrows: int | None = None,
    scope: str = "both",
) -> dict[str, Any]:
    logger = get_logger(__name__)
    config = load_data_config(config_path)
    scopes = _selected_scopes(scope)

    with tqdm(total=2 + (len(scopes) * 5), desc="Pipeline", unit="step") as progress:
        logger.info("Loading raw tables from %s", config.raw_dir)
        tables = load_raw_tables(
            config.raw_dir,
            include_champs=config.include_champs,
            nrows=nrows,
        )
        progress.update(1)

        scope_summaries: dict[str, Any] = {}
        for selected_scope in scopes:
            scope_summaries[selected_scope] = _run_scope_pipeline(
                selected_scope,
                tables=tables,
                config_path=config_path,
                config=config,
            )
            progress.update(5)

        summary = {
            "requested_scope": scope,
            "config": {
                "data_dir": str(config.data_dir),
                "raw_dir": str(config.raw_dir),
                "interim_dir": str(config.interim_dir),
                "processed_dir": str(config.processed_dir),
                "target_column": config.target_column,
                "group_column": config.group_column,
                "test_size": config.test_size,
                "val_size": config.val_size,
                "random_state": config.random_state,
                "include_champs": config.include_champs,
                "min_curated_duration_seconds": config.min_curated_duration_seconds,
            },
            "raw_tables": {
                name: {
                    "rows": int(len(frame)),
                    "columns": int(len(frame.columns)),
                }
                for name, frame in tables.items()
            },
            "scopes": scope_summaries,
        }

        summary_path = config.processed_dir / "pipeline_summary.json"
        _dump_json(summary_path, summary)
        logger.info("Wrote pipeline summary to %s", summary_path)
        progress.update(1)

    return summary


def main() -> None:
    args = parse_args()
    summary = build_pipeline(
        config_path=args.config, nrows=args.nrows, scope=args.scope
    )

    for scope_name, scope_summary in summary["scopes"].items():
        merged_rows = scope_summary["merged"]["rows"]
        train_rows = scope_summary["splits"]["train"]["rows"]
        val_rows = scope_summary["splits"]["val"]["rows"]
        test_rows = scope_summary["splits"]["test"]["rows"]
        print(
            f"{scope_name} pipeline complete: "
            f"merged={merged_rows:,}, train={train_rows:,}, val={val_rows:,}, test={test_rows:,}"
        )


if __name__ == "__main__":
    main()
