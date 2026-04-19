from __future__ import annotations

import pandas as pd

from ..utils.constants import (
    EXPECTED_TEAM_ROWS_PER_MATCH,
    TEAM_IDS,
    TEAM_KEY_COLUMNS,
)

BINARY_COLUMNS = [
    "win",
    "is_blue_side",
    "firstblood",
    "firsttower",
    "firstinhib",
    "firstbaron",
    "firstdragon",
    "firstharry",
]

NON_NEGATIVE_COLUMNS = [
    "kills",
    "deaths",
    "assists",
    "goldearned",
    "goldspent",
    "totdmgtochamp",
    "totminionskilled",
    "neutralminionskilled",
]


def _apply_team_level_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce team keys/flags, drop invalid rows, and keep only valid 2-team matches.

    Enforces unique (matchid, teamid) rows, team IDs in the allowed set,
    exactly two team rows per match, and exactly one winning team per match.
    """
    required = {"matchid", "teamid", "win", "is_blue_side"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Merged dataframe does not include required team-level columns: "
            + ", ".join(missing)
        )

    constrained = df.copy()
    constrained["matchid"] = pd.to_numeric(constrained["matchid"], errors="coerce")
    constrained["teamid"] = pd.to_numeric(constrained["teamid"], errors="coerce")
    constrained["win"] = (
        pd.to_numeric(constrained["win"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype("int8")
    )
    constrained["is_blue_side"] = (
        pd.to_numeric(constrained["is_blue_side"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype("int8")
    )

    constrained = constrained.dropna(subset=TEAM_KEY_COLUMNS).copy()
    constrained = constrained.loc[constrained["teamid"].isin(TEAM_IDS)].copy()

    duplicate_count = int(constrained.duplicated(subset=TEAM_KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            "Duplicate match/team keys found in merged dataframe: "
            f"duplicates={duplicate_count}."
        )

    teams_per_match = constrained.groupby("matchid")["teamid"].nunique(dropna=False)
    valid_matches = teams_per_match.loc[
        teams_per_match.eq(EXPECTED_TEAM_ROWS_PER_MATCH)
    ].index
    constrained = constrained.loc[constrained["matchid"].isin(valid_matches)].copy()

    wins_per_match = constrained.groupby("matchid")["win"].sum(min_count=1)
    valid_win_matches = wins_per_match.loc[wins_per_match.eq(1)].index
    constrained = constrained.loc[constrained["matchid"].isin(valid_win_matches)].copy()

    constrained = constrained.reset_index(drop=True)

    teams_per_match = constrained.groupby("matchid")["teamid"].nunique(dropna=False)
    wins_per_match = constrained.groupby("matchid")["win"].sum(min_count=1)
    if not teams_per_match.eq(EXPECTED_TEAM_ROWS_PER_MATCH).all():
        raise ValueError("Post-curation team rows per match are not exactly 2.")
    if not wins_per_match.eq(1).all():
        raise ValueError(
            "Post-curation winner constraints failed (sum(win) must be 1)."
        )

    return constrained


def clean_match_dataset(
    df: pd.DataFrame, min_duration_seconds: int | None = None
) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned.columns = [col.strip().lower() for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(0)

    object_cols = cleaned.select_dtypes(include=["object", "str"]).columns
    if len(object_cols) > 0:
        cleaned[object_cols] = cleaned[object_cols].fillna("unknown")

    for col in NON_NEGATIVE_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)
            cleaned[col] = cleaned[col].clip(lower=0)

    dynamic_non_negative = [
        col
        for col in numeric_cols
        if col not in {"matchid", "teamid", "win", "is_blue_side"}
    ]
    cleaned[dynamic_non_negative] = cleaned[dynamic_non_negative].clip(lower=0)

    for col in BINARY_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)
            cleaned[col] = cleaned[col].gt(0).astype("int8")

    # NOTE: we remove matches with very short time
    # abort-like matches from datasets
    if min_duration_seconds is not None and "duration" in cleaned.columns:
        cleaned["duration"] = pd.to_numeric(cleaned["duration"], errors="coerce")
        cleaned = cleaned.loc[cleaned["duration"].ge(min_duration_seconds)].reset_index(
            drop=True
        )

    cleaned = _apply_team_level_constraints(cleaned)

    return cleaned
