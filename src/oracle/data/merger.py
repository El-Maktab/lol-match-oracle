from __future__ import annotations

import pandas as pd


def _coerce_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def merge_match_level_dataset(
    matches: pd.DataFrame,
    participants: pd.DataFrame,
    stats: pd.DataFrame,
    teamstats: pd.DataFrame,
) -> pd.DataFrame:
    matches_ = _coerce_int(matches, ["id"])
    participants_ = _coerce_int(participants, ["id", "matchid", "player"])
    stats_ = _coerce_int(stats, ["id"])
    teamstats_ = _coerce_int(teamstats, ["matchid", "teamid"])

    merged = participants_.merge(
        stats_,
        on="id",
        how="inner",
        suffixes=("_participant", "_stat"),
    )

    merged = merged.merge(
        matches_,
        left_on="matchid",
        right_on="id",
        how="left",
        suffixes=("", "_match"),
    )

    # Infer team side from player slot for participant-level rows.
    merged["teamid"] = merged["player"].apply(
        lambda x: 100 if pd.notna(x) and int(x) <= 5 else 200
    )

    merged = merged.merge(
        teamstats_,
        on=["matchid", "teamid"],
        how="left",
        suffixes=("", "_team"),
    )

    return merged
