from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .champion_features import apply_champion_encoders, fit_champion_encoders
from .engineering import _assert_team_frame
from .pipeline import fit_transform_feature_splits


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


def _present_slot_columns(
    frame: pd.DataFrame, prefix: str, *, max_slots: int = 5
) -> list[str]:
    return [
        f"{prefix}_{index}"
        for index in range(1, max_slots + 1)
        if f"{prefix}_{index}" in frame.columns
    ]


def _fit_pick_rate_mapping(
    frame: pd.DataFrame, champion_cols: list[str]
) -> tuple[pd.Series, float]:
    if not champion_cols:
        return pd.Series(dtype="float64"), 0.0

    values = pd.concat(
        [pd.to_numeric(frame[col], errors="coerce") for col in champion_cols],
        axis=0,
        ignore_index=True,
    ).dropna()
    if values.empty:
        return pd.Series(dtype="float64"), 0.0

    pick_rates = values.value_counts(normalize=True)
    return pick_rates, float(pick_rates.mean())


def _fit_category_mapping(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}

    series = frame[column].astype("string").fillna("unknown")
    values = sorted(value for value in series.unique().tolist() if value is not None)
    return {value: index for index, value in enumerate(values)}


def _apply_category_mapping(
    frame: pd.DataFrame,
    *,
    column: str,
    output_column: str,
    mapping: dict[str, int],
) -> pd.DataFrame:
    out = frame.copy()
    if column not in out.columns:
        return out

    if pd.api.types.is_numeric_dtype(out[column]):
        out[output_column] = pd.to_numeric(out[column], errors="coerce").fillna(-1)
        return out

    out[output_column] = (
        out[column]
        .astype("string")
        .fillna("unknown")
        .map(mapping)
        .fillna(-1)
        .astype("int16")
    )
    return out


def _add_match_relative_feature(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    if column not in out.columns or "matchid" not in out.columns:
        return out

    numeric = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    opponent_value = numeric.groupby(out["matchid"]).transform("sum") - numeric
    out[f"{column}_diff_vs_opp"] = numeric - opponent_value
    return out


def _role_diversity(role_frame: pd.DataFrame) -> pd.Series:
    normalized = (
        role_frame.astype("string").fillna("unknown").apply(lambda col: col.str.lower())
    )
    return normalized.apply(
        lambda row: len({value for value in row.tolist() if value != "unknown"}),
        axis=1,
    )


def _add_pregame_features(
    frame: pd.DataFrame,
    *,
    champion_pick_rates: pd.Series,
    champion_pick_rate_prior: float,
    champion_prior: float,
    platform_mapping: dict[str, int],
    season_mapping: dict[str, int],
    target_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    champion_cols = _present_slot_columns(out, "champion")
    ban_cols = _present_slot_columns(out, "banned_champion")
    role_cols = _present_slot_columns(out, "role")

    for col in champion_cols:
        pick_rate_col = f"{col}_pick_rate"
        out[pick_rate_col] = (
            pd.to_numeric(out[col], errors="coerce")
            .map(champion_pick_rates)
            .fillna(champion_pick_rate_prior)
        )

    champion_te_cols = [
        f"{col}_winrate_te"
        for col in champion_cols
        if f"{col}_winrate_te" in out.columns
    ]
    champion_pick_rate_cols = [
        f"{col}_pick_rate" for col in champion_cols if f"{col}_pick_rate" in out.columns
    ]
    ban_te_cols = [
        f"{col}_winrate_te" for col in ban_cols if f"{col}_winrate_te" in out.columns
    ]

    if champion_te_cols:
        encoded = out[champion_te_cols].apply(pd.to_numeric, errors="coerce")
        out["team_avg_champ_wr"] = encoded.mean(axis=1)
        out["team_min_champ_wr"] = encoded.min(axis=1)
        out["team_max_champ_wr"] = encoded.max(axis=1)
        out["team_std_champ_wr"] = encoded.std(axis=1).fillna(0.0)
        out["team_champ_wr_span"] = out["team_max_champ_wr"] - out["team_min_champ_wr"]
    else:
        out["team_avg_champ_wr"] = champion_prior
        out["team_min_champ_wr"] = champion_prior
        out["team_max_champ_wr"] = champion_prior
        out["team_std_champ_wr"] = 0.0
        out["team_champ_wr_span"] = 0.0

    if champion_pick_rate_cols:
        pick_rate_frame = out[champion_pick_rate_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        out["team_avg_pick_rate"] = pick_rate_frame.mean(axis=1)
        out["team_max_pick_rate"] = pick_rate_frame.max(axis=1)
        out["team_min_pick_rate"] = pick_rate_frame.min(axis=1)
    else:
        out["team_avg_pick_rate"] = champion_pick_rate_prior
        out["team_max_pick_rate"] = champion_pick_rate_prior
        out["team_min_pick_rate"] = champion_pick_rate_prior

    if ban_te_cols:
        ban_frame = out[ban_te_cols].apply(pd.to_numeric, errors="coerce")
        out["avg_banned_champ_wr"] = ban_frame.mean(axis=1)
        out["max_banned_champ_wr"] = ban_frame.max(axis=1)

    out["draft_size"] = out[champion_cols].notna().sum(axis=1) if champion_cols else 0
    out["has_full_draft"] = (
        (out["draft_size"] == len(champion_cols)).astype("int8") if champion_cols else 0
    )
    out["unique_champion_count"] = (
        out[champion_cols].apply(lambda row: row.nunique(dropna=True), axis=1)
        if champion_cols
        else 0
    )

    if role_cols:
        role_frame = out[role_cols].astype("string").fillna("unknown")
        out["role_known_count"] = role_frame.ne("unknown").sum(axis=1)
        out["role_diversity"] = _role_diversity(role_frame)

    if "n_bans" in out.columns:
        out["n_bans"] = pd.to_numeric(out["n_bans"], errors="coerce").fillna(0)
        out["has_ban_data"] = out["n_bans"].gt(0).astype("int8")

    out = _apply_category_mapping(
        out,
        column="platformid",
        output_column="platform_code",
        mapping=platform_mapping,
    )
    out = _apply_category_mapping(
        out,
        column="seasonid",
        output_column="season_code",
        mapping=season_mapping,
    )

    for diff_col in ["team_avg_champ_wr", "team_max_champ_wr", "team_avg_pick_rate"]:
        out = _add_match_relative_feature(out, diff_col)

    if target_col not in out.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in pre-game feature frame."
        )

    return out


def build_pregame_feature_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    processed_dir: Path,
    target_col: str = "win",
    output_prefix: str = "pregame_",
) -> dict[str, Any]:
    _assert_team_frame(train_df, name="train split", target_col=target_col)
    _assert_team_frame(val_df, name="val split", target_col=target_col)
    _assert_team_frame(test_df, name="test split", target_col=target_col)

    champion_cols = _present_slot_columns(train_df, "champion")
    champion_artifacts = fit_champion_encoders(train_df, target_col=target_col)
    champion_pick_rates, champion_pick_rate_prior = _fit_pick_rate_mapping(
        train_df,
        champion_cols,
    )
    platform_mapping = _fit_category_mapping(train_df, "platformid")
    season_mapping = _fit_category_mapping(train_df, "seasonid")

    train_enriched = _add_pregame_features(
        apply_champion_encoders(train_df, champion_artifacts),
        champion_pick_rates=champion_pick_rates,
        champion_pick_rate_prior=champion_pick_rate_prior,
        champion_prior=champion_artifacts.global_rate,
        platform_mapping=platform_mapping,
        season_mapping=season_mapping,
        target_col=target_col,
    )
    val_enriched = _add_pregame_features(
        apply_champion_encoders(val_df, champion_artifacts),
        champion_pick_rates=champion_pick_rates,
        champion_pick_rate_prior=champion_pick_rate_prior,
        champion_prior=champion_artifacts.global_rate,
        platform_mapping=platform_mapping,
        season_mapping=season_mapping,
        target_col=target_col,
    )
    test_enriched = _add_pregame_features(
        apply_champion_encoders(test_df, champion_artifacts),
        champion_pick_rates=champion_pick_rates,
        champion_pick_rate_prior=champion_pick_rate_prior,
        champion_prior=champion_artifacts.global_rate,
        platform_mapping=platform_mapping,
        season_mapping=season_mapping,
        target_col=target_col,
    )

    selected_features = [
        column
        for column in [
            *[
                f"{col}_winrate_te"
                for col in champion_cols
                if f"{col}_winrate_te" in train_enriched.columns
            ],
            *[
                f"{col}_pick_rate"
                for col in champion_cols
                if f"{col}_pick_rate" in train_enriched.columns
            ],
            "team_avg_champ_wr",
            "team_min_champ_wr",
            "team_max_champ_wr",
            "team_std_champ_wr",
            "team_champ_wr_span",
            "team_avg_pick_rate",
            "team_max_pick_rate",
            "team_min_pick_rate",
            "team_avg_champ_wr_diff_vs_opp",
            "team_max_champ_wr_diff_vs_opp",
            "team_avg_pick_rate_diff_vs_opp",
            "avg_banned_champ_wr",
            "max_banned_champ_wr",
            "draft_size",
            "has_full_draft",
            "unique_champion_count",
            "role_known_count",
            "role_diversity",
            "n_bans",
            "has_ban_data",
            "is_blue_side",
            "platform_code",
            "season_code",
        ]
        if column in train_enriched.columns
    ]

    train_features, val_features, test_features, pipeline_summary, preprocessor = (
        fit_transform_feature_splits(
            train_enriched,
            val_enriched,
            test_enriched,
            feature_cols=selected_features,
            target_col=target_col,
            id_cols=["matchid", "teamid"],
            min_features_after_vif=5,
        )
    )

    train_out = processed_dir / f"{output_prefix}train_features.csv.gz"
    val_out = processed_dir / f"{output_prefix}val_features.csv.gz"
    test_out = processed_dir / f"{output_prefix}test_features.csv.gz"
    preprocessor_out = processed_dir / f"{output_prefix}feature_preprocessor.pkl"
    summary_out = processed_dir / f"{output_prefix}feature_engineering_summary.json"

    _write_csv_gz(train_features, train_out)
    _write_csv_gz(val_features, val_out)
    _write_csv_gz(test_features, test_out)
    _dump_pickle(preprocessor_out, preprocessor)

    summary: dict[str, Any] = {
        "scope": "pre-game draft-only features",
        "inputs": {
            "train": str(processed_dir / f"{output_prefix}train.csv.gz"),
            "val": str(processed_dir / f"{output_prefix}val.csv.gz"),
            "test": str(processed_dir / f"{output_prefix}test.csv.gz"),
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
        "module_summary": {
            "champion_encoding_columns": sorted(champion_artifacts.mappings.keys()),
            "champion_encoding_prior": champion_artifacts.global_rate,
            "champion_pick_rate_prior": champion_pick_rate_prior,
            "platform_levels": len(platform_mapping),
            "season_levels": len(season_mapping),
        },
        "rows": {
            "train": int(len(train_features)),
            "val": int(len(val_features)),
            "test": int(len(test_features)),
        },
    }

    _dump_json(summary_out, summary)
    return summary
