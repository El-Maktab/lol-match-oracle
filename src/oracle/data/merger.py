"""Build a team-level match dataset from raw LoL tables.

What this module does
-----------------------------------
1. Joins participant rows with player stats using participant id.
2. Infers team side (100 blue, 200 red) from player slot when teamid is missing.
3. Aggregates participant numeric features to one row per (matchid, teamid).
4. Derives a team win label from participant-level win values.
5. Adds team objective stats and match metadata.
6. Adds an `is_blue_side` binary flag.

Expected inputs
---------------
- matches:
    - Required: `id` (match id)
- participants:
    - Required: `id`, `matchid`, `player`
- stats:
    - Required: `id`, `win`
- teamstats:
    - Required: `matchid`, `teamid`

Expected output
---------------
Returns a pandas DataFrame with one row per team in each match that survives
the joins and filtering (`teamid` in [100, 200]).

Output columns are produced in these groups:
- Team keys:
    - `matchid`, `teamid`
- Aggregated participant features:
    - For each participant numeric feature `x`: `x_sum`, `x_mean`
    - For each participant binary feature `b` in {0,1}: `b_rate`
- Team outcome quality columns:
    - `win` (majority participant win, coerced to 0/1)
    - `participant_win_consistency` (majority share in [0,1])
    - `participant_win_inconsistent` (1 if participant win values disagree)
- Teamstats columns:
    - All columns from `teamstats` (deduplicated on `matchid`, `teamid`)
- Match metadata columns:
    - All columns from `matches` with `id` renamed to `matchid`
- Side indicator:
    - `is_blue_side` (1 for blue team id 100, else 0)
"""

from __future__ import annotations

import pandas as pd

from ..utils.constants import BLUE_TEAM_ID, RED_TEAM_ID, TEAM_IDS, TEAM_KEY_COLUMNS


def _coerce_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _infer_team_id(player_slot: object) -> int | None:
    if pd.isna(player_slot):
        return None
    slot = int(player_slot)
    return BLUE_TEAM_ID if slot <= 5 else RED_TEAM_ID


def _derive_team_win(participant_team_df: pd.DataFrame) -> pd.DataFrame:
    win_summary = (
        participant_team_df.groupby(TEAM_KEY_COLUMNS, as_index=False)["win"]
        .agg(["sum", "count", "mean", "nunique"])
        .reset_index()
    )
    win_summary.columns = [
        col if not isinstance(col, tuple) else col[0] if col[1] == "" else col[1]
        for col in win_summary.columns
    ]

    wins = pd.to_numeric(win_summary["sum"], errors="coerce").fillna(0)
    sample_size = pd.to_numeric(win_summary["count"], errors="coerce").fillna(0)
    majority_votes = wins.combine(sample_size - wins, max)

    win_summary["participant_win_consistency"] = majority_votes / sample_size.clip(
        lower=1
    )
    win_summary["participant_win_inconsistent"] = (
        pd.to_numeric(win_summary["nunique"], errors="coerce")
        .fillna(0)
        .gt(1)
        .astype("int8")
    )
    win_summary["win"] = (
        pd.to_numeric(win_summary["mean"], errors="coerce")
        .fillna(0)
        .ge(0.5)
        .astype("int8")
    )

    return win_summary[
        TEAM_KEY_COLUMNS
        + ["win", "participant_win_consistency", "participant_win_inconsistent"]
    ]


def _aggregate_participant_features(participant_team_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = participant_team_df.select_dtypes(include=["number"]).columns
    excluded_cols = {
        "id",
        "matchid",
        "player",
        "teamid",
        "ss1",
        "ss2",
        "championid",
        "win",
    }

    feature_cols = [col for col in numeric_cols if col not in excluded_cols]
    if not feature_cols:
        raise ValueError(
            "No numeric participant-level features available for aggregation."
        )

    grouped = participant_team_df.groupby(TEAM_KEY_COLUMNS, as_index=False)
    aggregated_sum = grouped[feature_cols].sum().add_suffix("_sum")
    aggregated_mean = grouped[feature_cols].mean().add_suffix("_mean")
    aggregated = aggregated_sum.merge(
        aggregated_mean, left_index=True, right_index=True
    )

    rename_key_cols = {
        "matchid_sum": "matchid",
        "teamid_sum": "teamid",
    }
    aggregated = aggregated.rename(columns=rename_key_cols)

    extra_key_cols = ["matchid_mean", "teamid_mean"]
    drop_key_cols = [col for col in extra_key_cols if col in aggregated.columns]
    if drop_key_cols:
        aggregated = aggregated.drop(columns=drop_key_cols)

    binary_cols: list[str] = []
    for col in feature_cols:
        series = pd.to_numeric(participant_team_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        if set(series.unique().tolist()).issubset({0, 1}):
            binary_cols.append(col)

    for col in binary_cols:
        rates = (
            participant_team_df.groupby(TEAM_KEY_COLUMNS, as_index=False)[col]
            .mean()
            .rename(columns={col: f"{col}_rate"})
        )
        aggregated = aggregated.merge(rates, on=TEAM_KEY_COLUMNS, how="left")

    team_win = _derive_team_win(participant_team_df)
    return aggregated.merge(team_win, on=TEAM_KEY_COLUMNS, how="left")


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

    participant_merged = participants_.merge(
        stats_,
        on="id",
        how="inner",
        suffixes=("_participant", "_stat"),
    )

    inferred_team = participant_merged["player"].apply(_infer_team_id)
    if "teamid" in participant_merged.columns:
        participant_merged["teamid"] = (
            pd.to_numeric(participant_merged["teamid"], errors="coerce")
            .astype("Int64")
            .fillna(inferred_team)
        )
    else:
        participant_merged["teamid"] = inferred_team

    participant_merged["win"] = (
        pd.to_numeric(participant_merged["win"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype("int8")
    )

    participant_merged = participant_merged.dropna(subset=TEAM_KEY_COLUMNS).copy()
    participant_merged["teamid"] = participant_merged["teamid"].astype("Int64")
    participant_merged = participant_merged.loc[
        participant_merged["teamid"].isin(TEAM_IDS)
    ].copy()

    participant_team = _aggregate_participant_features(participant_merged)

    matches_by_team = matches_.rename(columns={"id": "matchid"})
    teamstats_features = teamstats_.drop_duplicates(subset=TEAM_KEY_COLUMNS)

    merged = participant_team.merge(
        teamstats_features,
        on=TEAM_KEY_COLUMNS,
        how="left",
        suffixes=("", "_team"),
    )

    merged = merged.merge(
        matches_by_team,
        on="matchid",
        how="left",
        suffixes=("", "_match"),
    )

    merged["is_blue_side"] = (merged["teamid"] == BLUE_TEAM_ID).astype("int8")

    return merged
