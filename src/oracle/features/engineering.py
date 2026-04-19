from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .champion_features import apply_champion_encoders, fit_champion_encoders
from .pipeline import fit_transform_feature_splits
from .player_features import add_player_features
from .team_features import add_team_features
from ..utils.constants import (
    EXPECTED_TEAM_ROWS_PER_MATCH,
    TEAM_IDS,
    TEAM_KEY_COLUMNS,
)


def _assert_team_frame(df: pd.DataFrame, *, name: str, target_col: str) -> None:
    # NOTE: Feature engineering only runs on team-level, two-rows-per-match curated inputs.
    required = {"matchid", "teamid", target_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")

    key_dupes = int(df.duplicated(subset=TEAM_KEY_COLUMNS).sum())
    if key_dupes:
        raise ValueError(f"{name} contains duplicate match/team rows: {key_dupes}")

    if not pd.to_numeric(df["teamid"], errors="coerce").isin(TEAM_IDS).all():
        raise ValueError(f"{name} contains invalid teamid values.")

    teams_per_match = df.groupby("matchid")["teamid"].nunique(dropna=False)
    if not teams_per_match.eq(EXPECTED_TEAM_ROWS_PER_MATCH).all():
        raise ValueError(f"{name} does not have exactly 2 team rows for every match.")

    wins_per_match = (
        pd.to_numeric(df[target_col], errors="coerce").groupby(df["matchid"]).sum()
    )
    if not wins_per_match.eq(1).all():
        raise ValueError(
            f"{name} violates winner constraint (sum(win) must equal 1 per match)."
        )


def _build_enriched_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_enriched = add_team_features(add_player_features(train_df))
    val_enriched = add_team_features(add_player_features(val_df))
    test_enriched = add_team_features(add_player_features(test_df))

    # NOTE: Champion encoders are fit on train only, then applied to val/test to prevent leakage.
    champion_artifacts = fit_champion_encoders(train_enriched, target_col=target_col)
    train_enriched = apply_champion_encoders(train_enriched, champion_artifacts)
    val_enriched = apply_champion_encoders(val_enriched, champion_artifacts)
    test_enriched = apply_champion_encoders(test_enriched, champion_artifacts)

    module_summary = {
        "champion_encoding_columns": sorted(champion_artifacts.mappings.keys()),
        "champion_encoding_prior": champion_artifacts.global_rate,
    }

    return train_enriched, val_enriched, test_enriched, module_summary


def _select_candidate_features(frame: pd.DataFrame, *, target_col: str) -> list[str]:
    id_cols = ["matchid", "teamid"]
    # NOTE: Reserved metadata columns are excluded even if numeric.
    reserved = {
        *id_cols,
        target_col,
        "gameid",
        "queueid",
        "seasonid",
        "creation",
        "duration",
    }

    numeric_features = frame.select_dtypes(include=["number"]).columns.tolist()

    def _is_target_derived(col: str) -> bool:
        # Guardrail: exclude columns directly or indirectly built from target labels.
        normalized = col.lower()

        target_linked_exact = {
            "participant_win_consistency",
            "participant_win_inconsistent",
            "champion_signal_prior",
        }
        if normalized in target_linked_exact:
            return True

        # Block full families that are derived from win labels or target-encoding stats.
        blocked_substrings = (
            "participant_win",
            "_winrate_te",
            "target_encoding",
            "label_encoding",
        )
        if any(token in normalized for token in blocked_substrings):
            return True

        # Also block generic target/label naming patterns.
        if (
            normalized.startswith("target_")
            or normalized.endswith("_target")
            or normalized.startswith("label_")
            or normalized.endswith("_label")
        ):
            return True

        if (
            normalized.startswith(f"{target_col}_")
            or normalized.endswith(f"_{target_col}")
            or f"_{target_col}_" in normalized
        ):
            return True

        return False

    return [
        col for col in numeric_features if col not in reserved and not _is_target_derived(col)
    ]


def _write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def build_feature_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    processed_dir: Path,
    target_col: str = "win",
) -> dict[str, Any]:
    _assert_team_frame(train_df, name="train split", target_col=target_col)
    _assert_team_frame(val_df, name="val split", target_col=target_col)
    _assert_team_frame(test_df, name="test split", target_col=target_col)

    train_enriched, val_enriched, test_enriched, module_summary = (
        _build_enriched_splits(
            train_df,
            val_df,
            test_df,
            target_col=target_col,
        )
    )

    candidate_features = _select_candidate_features(
        train_enriched,
        target_col=target_col,
    )

    train_features, val_features, test_features, pipeline_summary, preprocessor = (
        fit_transform_feature_splits(
            train_enriched,
            val_enriched,
            test_enriched,
            feature_cols=candidate_features,
            target_col=target_col,
            id_cols=["matchid", "teamid"],
        )
    )

    train_out = processed_dir / "train_features.csv.gz"
    val_out = processed_dir / "val_features.csv.gz"
    test_out = processed_dir / "test_features.csv.gz"
    preprocessor_out = processed_dir / "feature_preprocessor.pkl"
    summary_out = processed_dir / "feature_engineering_summary.json"

    _write_csv_gz(train_features, train_out)
    _write_csv_gz(val_features, val_out)
    _write_csv_gz(test_features, test_out)
    # NOTE: Persisting the fitted preprocessor keeps training and serving transforms aligned.
    _dump_pickle(preprocessor_out, preprocessor)

    summary: dict[str, Any] = {
        "scope": "post-game analysis with team-aggregated features",
        "inputs": {
            "train": str(processed_dir / "train.csv.gz"),
            "val": str(processed_dir / "val.csv.gz"),
            "test": str(processed_dir / "test.csv.gz"),
        },
        "outputs": {
            "train_features": str(train_out),
            "val_features": str(val_out),
            "test_features": str(test_out),
            "preprocessor": str(preprocessor_out),
        },
        "feature_counts": {
            "candidate": int(pipeline_summary["candidate_feature_count"]),
            "after_correlation": int(pipeline_summary["after_correlation"]),
            "after_vif": int(pipeline_summary["after_vif"]),
            "final_selected": int(pipeline_summary["final_selected"]),
        },
        "selected_features": pipeline_summary["selected_features"],
        "multicollinearity": {
            "dropped_high_correlation": pipeline_summary["dropped_high_correlation"],
            "dropped_high_vif": pipeline_summary["dropped_high_vif"],
            "pca_components_95pct": pipeline_summary["pca_components_95pct"],
            "vif_top": pipeline_summary["vif_top"],
        },
        "transformations": {
            "binary_passthrough_cols": pipeline_summary["binary_passthrough_cols"],
            "robust_scaled_cols": pipeline_summary["robust_scaled_cols"],
            "standard_scaled_cols": pipeline_summary["standard_scaled_cols"],
        },
        "module_summary": module_summary,
        "rows": {
            "train": int(len(train_features)),
            "val": int(len(val_features)),
            "test": int(len(test_features)),
        },
        "team_level_constraints": {
            "required_keys": TEAM_KEY_COLUMNS,
            "team_ids": TEAM_IDS,
            "expected_rows_per_match": EXPECTED_TEAM_ROWS_PER_MATCH,
            "winner_rule": "sum(win)==1 per match",
        },
    }

    # NOTE: This summary is the concise audit trail for feature decisions and outputs.
    _dump_json(summary_out, summary)
    return summary
