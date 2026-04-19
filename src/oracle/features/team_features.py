"""Team-level feature engineering from merged match rows.

This module builds team context features that capture objective control,
opponent-relative advantages, and pace-aware gold pressure signals.

Main outputs include:
- Side indicator: `is_blue_side`.
- Objective features: `objective_control_score`, `objective_control_rate`.
- Opponent-relative deltas: `<feature>_diff_vs_opp` for key combat/economy stats.
- Gold pressure proxies: `gold_diff_per_min`, `gold_adv_15_proxy`,
  `gold_adv_25_proxy`.
- Team fight load proxy: `kill_activity`.
"""

from __future__ import annotations

import pandas as pd


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(num, errors="coerce").fillna(0)
    denominator = pd.to_numeric(den, errors="coerce").fillna(0).clip(lower=1e-6)
    return numerator / denominator


def _first_available(df: pd.DataFrame, options: list[str]) -> str | None:
    for col in options:
        if col in df.columns:
            return col
    return None


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create team objective and opponent-relative features."""
    out = df.copy()

    if "is_blue_side" not in out.columns and "teamid" in out.columns:
        # NOTE: Keep side as explicit binary because it captures stable map asymmetry.
        out["is_blue_side"] = (
            pd.to_numeric(out["teamid"], errors="coerce") == 100
        ).astype("int8")

    objective_cols = [
        col
        for col in [
            "towerkills",
            "inhibkills",
            "dragonkills",
            "baronkills",
            "harrykills",
        ]
        if col in out.columns
    ]
    if objective_cols:
        # NOTE: Additive objective score is an intentional compact proxy for objective tempo.
        out["objective_control_score"] = out[objective_cols].sum(axis=1)
        minutes = pd.to_numeric(out.get("match_minutes", 0), errors="coerce").fillna(0)
        if not minutes.gt(0).any() and "duration" in out.columns:
            minutes = pd.to_numeric(out["duration"], errors="coerce").fillna(0) / 60.0
        # NOTE: Objective control rate normalizes objective progress by game length.
        out["objective_control_rate"] = _safe_ratio(
            out["objective_control_score"], minutes.clip(lower=1)
        )

    relative_feature_candidates = [
        "kills_sum",
        "assists_sum",
        "deaths_sum",
        "goldearned_sum",
        "totdmgtochamp_sum",
        "visionscore_sum",
        "towerkills",
        "dragonkills",
        "baronkills",
        "inhibkills",
    ]

    if {"matchid", "teamid"}.issubset(out.columns):
        for col in relative_feature_candidates:
            if col not in out.columns:
                continue
            # NOTE: Opponent-relative deltas are computed within-match to avoid cross-match leakage.
            numeric_col = pd.to_numeric(out[col], errors="coerce").fillna(0)
            match_totals = numeric_col.groupby(out["matchid"]).transform("sum")
            opp_value = match_totals - numeric_col
            out[f"{col}_diff_vs_opp"] = numeric_col - opp_value

    gold_col = _first_available(
        out, ["goldearned_sum_diff_vs_opp", "goldearned_diff_vs_opp"]
    )
    if gold_col and "match_minutes" in out.columns:
        # NOTE: 15/25 proxies assume near-linear gold pace when true timeline gold is unavailable.
        # NOTE: Gold diff per minute captures the speed of economic advantage growth.
        out["gold_diff_per_min"] = _safe_ratio(out[gold_col], out["match_minutes"])
        out["gold_adv_15_proxy"] = out["gold_diff_per_min"] * 15.0
        out["gold_adv_25_proxy"] = out["gold_diff_per_min"] * 25.0

    kills_col = _first_available(out, ["kills_sum", "kills"])
    assists_col = _first_available(out, ["assists_sum", "assists"])
    if kills_col and assists_col:
        # NOTE: Kill activity approximates total fight participation pressure.
        out["kill_activity"] = pd.to_numeric(out[kills_col], errors="coerce").fillna(
            0
        ) + pd.to_numeric(out[assists_col], errors="coerce").fillna(0)

    return out
