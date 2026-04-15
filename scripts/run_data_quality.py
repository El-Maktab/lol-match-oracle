from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import great_expectations as gx
import pandas as pd

from oracle.data import clean_match_dataset, load_raw_tables, merge_match_level_dataset
from oracle.utils import get_logger, load_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up Great Expectations suites and run initial checkpoints."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to data config file (defaults to configs/data.yaml).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for quick local validation.",
    )
    return parser.parse_args()


def _ensure_batch_definition(asset: Any, batch_definition_name: str) -> Any:
    try:
        return asset.get_batch_definition(batch_definition_name)
    except Exception:
        return asset.add_batch_definition_whole_dataframe(batch_definition_name)


def _prepare_validator(validator: Any) -> Any:
    # GE sets result_format at validator defaults (BASIC), which triggers
    # a persistence warning when adding expectations in v1.16.
    validator.default_expectation_args.pop("result_format", None)
    return validator


def _add_raw_matches_expectations(validator: Any, matches_df: pd.DataFrame) -> None:
    expected_columns = [
        "id",
        "gameid",
        "platformid",
        "queueid",
        "seasonid",
        "duration",
        "creation",
        "version",
    ]

    validator.expect_table_columns_to_match_set(expected_columns)
    validator.expect_column_values_to_not_be_null("id")
    validator.expect_column_values_to_be_unique("id")
    validator.expect_column_values_to_not_be_null("gameid")
    validator.expect_column_values_to_be_between(
        "duration", min_value=0, max_value=7200
    )

    valid_season_ids = (
        pd.to_numeric(matches_df["seasonid"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    validator.expect_column_values_to_be_in_set("seasonid", valid_season_ids)


def _add_raw_stats_expectations(validator: Any) -> None:
    required_columns = [
        "id",
        "win",
        "kills",
        "deaths",
        "assists",
        "goldearned",
        "totdmgtochamp",
    ]

    for col in required_columns:
        validator.expect_column_to_exist(col)

    validator.expect_column_values_to_not_be_null("id")
    validator.expect_column_values_to_be_unique("id")
    validator.expect_column_values_to_be_in_set("win", [0, 1])

    for col in ["kills", "deaths", "assists", "goldearned", "totdmgtochamp"]:
        validator.expect_column_values_to_be_between(col, min_value=0)


def _add_processed_expectations(validator: Any, processed_df: pd.DataFrame) -> None:
    validator.expect_column_to_exist("win")
    validator.expect_column_values_to_be_in_set("win", [0, 1])

    for col in processed_df.columns:
        validator.expect_column_values_to_not_be_null(col)

    leakage_columns = [
        "winner",
        "result",
        "bluewins",
    ]
    present_leakage = [col for col in leakage_columns if col in processed_df.columns]
    if present_leakage:
        raise ValueError(
            "Data leakage columns found in processed features: "
            + ", ".join(present_leakage)
        )


def _export_expectation_suite(context: Any, suite_name: str, output_path: Path) -> None:
    suite = context.suites.get(name=suite_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(suite.to_json_dict(), handle, indent=2)
        handle.write("\n")


def _run_checkpoint(
    context: Any,
    checkpoint_name: str,
    validation_name: str,
    batch_definition: Any,
    suite_name: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    suite = context.suites.get(name=suite_name)

    validation_definition = gx.ValidationDefinition(
        name=validation_name,
        data=batch_definition,
        suite=suite,
    )
    validation_definition = context.validation_definitions.add_or_update(
        validation_definition
    )

    checkpoint = gx.Checkpoint(
        name=checkpoint_name,
        validation_definitions=[validation_definition],
    )
    checkpoint = context.checkpoints.add_or_update(checkpoint)

    result = checkpoint.run(batch_parameters={"dataframe": dataframe})
    return {
        "checkpoint": checkpoint_name,
        "success": bool(result.success),
        "run_results": int(len(result.run_results)),
    }


def run_data_quality(
    config_path: Path | None = None, nrows: int | None = None
) -> dict[str, Any]:
    logger = get_logger(__name__)
    config = load_data_config(config_path)

    logger.info("Loading source data")
    tables = load_raw_tables(
        config.raw_dir, include_champs=config.include_champs, nrows=nrows
    )

    logger.info("Building merged dataframe")
    merged_df = merge_match_level_dataset(
        tables["matches"],
        tables["participants"],
        tables["stats"],
        tables["teamstats"],
    )
    merged_df = clean_match_dataset(
        merged_df,
        min_duration_seconds=config.min_curated_duration_seconds,
    )

    logger.info("Loading processed splits")
    processed_train_path = config.processed_dir / "train.csv.gz"
    if not processed_train_path.exists():
        raise FileNotFoundError(
            "Missing processed split: data/processed/train.csv.gz. Run scripts/run_pipeline.py first."
        )
    processed_train_df = pd.read_csv(processed_train_path, nrows=nrows)

    logger.info("Initializing Great Expectations file context")
    context = gx.get_context(mode="file", project_root_dir="great_expectations")

    data_source = context.data_sources.add_or_update_pandas(name="lol_match_oracle")

    raw_matches_asset = (
        data_source.get_asset("raw_matches")
        if "raw_matches" in data_source.get_asset_names()
        else data_source.add_dataframe_asset("raw_matches")
    )
    raw_stats_asset = (
        data_source.get_asset("raw_stats")
        if "raw_stats" in data_source.get_asset_names()
        else data_source.add_dataframe_asset("raw_stats")
    )
    merged_asset = (
        data_source.get_asset("merged_match_level")
        if "merged_match_level" in data_source.get_asset_names()
        else data_source.add_dataframe_asset("merged_match_level")
    )
    processed_asset = (
        data_source.get_asset("processed_train")
        if "processed_train" in data_source.get_asset_names()
        else data_source.add_dataframe_asset("processed_train")
    )

    raw_matches_bd = _ensure_batch_definition(raw_matches_asset, "raw_matches_full")
    raw_stats_bd = _ensure_batch_definition(raw_stats_asset, "raw_stats_full")
    merged_bd = _ensure_batch_definition(merged_asset, "merged_match_level_full")
    processed_bd = _ensure_batch_definition(processed_asset, "processed_train_full")

    logger.info("Defining expectation suites")
    raw_matches_suite = context.suites.add_or_update(
        gx.ExpectationSuite(name="raw_matches_suite")
    )
    raw_stats_suite = context.suites.add_or_update(
        gx.ExpectationSuite(name="raw_stats_suite")
    )
    processed_suite = context.suites.add_or_update(
        gx.ExpectationSuite(name="processed_features_suite")
    )

    raw_matches_validator = _prepare_validator(
        context.get_validator(
            batch_request=raw_matches_bd.build_batch_request(
                batch_parameters={"dataframe": tables["matches"]}
            ),
            expectation_suite=raw_matches_suite,
        )
    )
    _add_raw_matches_expectations(raw_matches_validator, tables["matches"])
    context.suites.add_or_update(raw_matches_validator.expectation_suite)

    raw_stats_validator = _prepare_validator(
        context.get_validator(
            batch_request=raw_stats_bd.build_batch_request(
                batch_parameters={"dataframe": tables["stats"]}
            ),
            expectation_suite=raw_stats_suite,
        )
    )
    _add_raw_stats_expectations(raw_stats_validator)
    context.suites.add_or_update(raw_stats_validator.expectation_suite)

    processed_validator = _prepare_validator(
        context.get_validator(
            batch_request=processed_bd.build_batch_request(
                batch_parameters={"dataframe": processed_train_df}
            ),
            expectation_suite=processed_suite,
        )
    )
    _add_processed_expectations(processed_validator, processed_train_df)
    context.suites.add_or_update(processed_validator.expectation_suite)

    # Merged dataset contract extends the processed suite checks to ensure merged integrity.
    merged_suite = context.suites.add_or_update(
        gx.ExpectationSuite(name="merged_match_level_suite")
    )
    merged_validator = _prepare_validator(
        context.get_validator(
            batch_request=merged_bd.build_batch_request(
                batch_parameters={"dataframe": merged_df}
            ),
            expectation_suite=merged_suite,
        )
    )
    merged_validator.expect_column_to_exist("matchid")
    merged_validator.expect_column_to_exist("teamid")
    merged_validator.expect_column_values_to_not_be_null("matchid")
    merged_validator.expect_column_values_to_not_be_null("teamid")
    merged_validator.expect_column_values_to_be_in_set("teamid", [100, 200])
    merged_validator.expect_column_values_to_be_in_set("win", [0, 1])
    merged_validator.expect_column_values_to_be_between(
        "duration", min_value=config.min_curated_duration_seconds, max_value=7200
    )
    context.suites.add_or_update(merged_validator.expectation_suite)

    logger.info("Exporting suite JSON files")
    expectations_dir = Path("great_expectations") / "expectations"
    _export_expectation_suite(
        context, "raw_matches_suite", expectations_dir / "raw_matches_suite.json"
    )
    _export_expectation_suite(
        context, "raw_stats_suite", expectations_dir / "raw_stats_suite.json"
    )
    _export_expectation_suite(
        context,
        "processed_features_suite",
        expectations_dir / "processed_features_suite.json",
    )

    logger.info("Running initial checkpoints")
    checkpoint_results = [
        _run_checkpoint(
            context,
            checkpoint_name="raw_matches_checkpoint",
            validation_name="raw_matches_validation",
            batch_definition=raw_matches_bd,
            suite_name="raw_matches_suite",
            dataframe=tables["matches"],
        ),
        _run_checkpoint(
            context,
            checkpoint_name="raw_stats_checkpoint",
            validation_name="raw_stats_validation",
            batch_definition=raw_stats_bd,
            suite_name="raw_stats_suite",
            dataframe=tables["stats"],
        ),
        _run_checkpoint(
            context,
            checkpoint_name="processed_features_checkpoint",
            validation_name="processed_features_validation",
            batch_definition=processed_bd,
            suite_name="processed_features_suite",
            dataframe=processed_train_df,
        ),
        _run_checkpoint(
            context,
            checkpoint_name="merged_match_level_checkpoint",
            validation_name="merged_match_level_validation",
            batch_definition=merged_bd,
            suite_name="merged_match_level_suite",
            dataframe=merged_df,
        ),
    ]

    logger.info("Building data docs")
    data_docs_sites = context.build_data_docs()

    report = {
        "rows": {
            "raw_matches": int(len(tables["matches"])),
            "raw_stats": int(len(tables["stats"])),
            "merged": int(len(merged_df)),
            "processed_train": int(len(processed_train_df)),
        },
        "raw_monitoring": {
            "short_duration_threshold_seconds": config.min_curated_duration_seconds,
            "short_duration_rows": int(
                pd.to_numeric(tables["matches"]["duration"], errors="coerce")
                .lt(config.min_curated_duration_seconds)
                .sum()
            ),
        },
        "checkpoint_results": checkpoint_results,
        "all_checkpoints_passed": all(item["success"] for item in checkpoint_results),
        "data_docs": {name: str(url) for name, url in data_docs_sites.items()},
    }

    report_path = config.processed_dir / "data_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Data quality report written to %s", report_path)

    return report


def main() -> None:
    args = parse_args()
    report = run_data_quality(config_path=args.config, nrows=args.nrows)

    print("Data quality checkpoints:")
    for row in report["checkpoint_results"]:
        status = "PASS" if row["success"] else "FAIL"
        print(f"- {row['checkpoint']}: {status}")
    print(f"all_checkpoints_passed={report['all_checkpoints_passed']}")


if __name__ == "__main__":
    main()
