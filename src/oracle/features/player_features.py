"""Player-derived feature engineering for team-level match rows.

This module creates pace-normalized and efficiency-style features from
participant-derived aggregates. The goal is to turn raw totals into signals that
are easier for models to compare across games with different durations and
tempos.

Main outputs include:
- Time context features (`match_minutes`, game phase flags).
- Combat efficiency (`kda_ratio`, `damage_per_gold`).
- Economy efficiency (`gold_spent_ratio`).
- Tempo-normalized activity (`objective_damage_per_min`, `cs_per_min`,
  `vision_per_min`, `cc_per_min`).
- Jungle pressure/control proxy (`jungle_control_share`).
"""

from __future__ import annotations

import pandas as pd


def _pick_column(df: pd.DataFrame, options: list[str]) -> str | None:
    for col in options:
        if col in df.columns:
            return col
    return None


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(num, errors="coerce").fillna(0)
    denominator = pd.to_numeric(den, errors="coerce").fillna(0).clip(lower=1e-6)
    return numerator / denominator


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create player-derived team-level features from aggregated split data."""
    out = df.copy()

    # NOTE: We support both *_sum aggregated columns and raw names for notebook/script parity.
    kills_col = _pick_column(out, ["kills_sum", "kills"])
    deaths_col = _pick_column(out, ["deaths_sum", "deaths"])
    assists_col = _pick_column(out, ["assists_sum", "assists"])
    earned_col = _pick_column(out, ["goldearned_sum", "goldearned"])
    spent_col = _pick_column(out, ["goldspent_sum", "goldspent"])
    damage_col = _pick_column(out, ["totdmgtochamp_sum", "totdmgtochamp"])
    objective_damage_col = _pick_column(out, ["dmgtoobj_sum", "dmgtoobj"])
    turret_damage_col = _pick_column(out, ["dmgtoturrets_sum", "dmgtoturrets"])
    minions_col = _pick_column(out, ["totminionskilled_sum", "totminionskilled"])
    neutral_col = _pick_column(
        out, ["neutralminionskilled_sum", "neutralminionskilled"]
    )
    vision_col = _pick_column(out, ["visionscore_sum", "visionscore"])
    cc_col = _pick_column(out, ["timecc_sum", "timecc"])
    own_jungle_col = _pick_column(out, ["ownjunglekills_sum", "ownjunglekills"])
    enemy_jungle_col = _pick_column(out, ["enemyjunglekills_sum", "enemyjunglekills"])

    # NOTE: Convert duration to minutes to make per-minute rates comparable across games.
    duration = pd.to_numeric(out.get("duration", 0), errors="coerce").fillna(0)
    out["match_minutes"] = duration.clip(lower=60) / 60.0

    # NOTE: Phase buckets are duration-based proxies because no timeline snapshots are available.
    out["phase_early"] = out["match_minutes"].lt(20).astype("int8")
    out["phase_mid"] = (
        out["match_minutes"].ge(20) & out["match_minutes"].lt(30)
    ).astype("int8")
    out["phase_late"] = out["match_minutes"].ge(30).astype("int8")

    if kills_col and assists_col and deaths_col:
        # NOTE: KDA captures fight efficiency by rewarding kills/assists and penalizing deaths.
        out["kda_ratio"] = _safe_ratio(
            pd.to_numeric(out[kills_col], errors="coerce").fillna(0)
            + pd.to_numeric(out[assists_col], errors="coerce").fillna(0),
            pd.to_numeric(out[deaths_col], errors="coerce").fillna(0).clip(lower=1),
        )

    if earned_col and spent_col:
        # NOTE: Higher spend ratio suggests stronger gold conversion into items.
        out["gold_spent_ratio"] = _safe_ratio(out[spent_col], out[earned_col])

    if damage_col and earned_col:
        # NOTE: Damage per gold is a compact efficiency signal for combat impact.
        out["damage_per_gold"] = _safe_ratio(out[damage_col], out[earned_col])

    if objective_damage_col and turret_damage_col:
        # NOTE: Objective pressure per minute approximates map progress tempo.
        out["objective_damage_per_min"] = _safe_ratio(
            pd.to_numeric(out[objective_damage_col], errors="coerce").fillna(0)
            + pd.to_numeric(out[turret_damage_col], errors="coerce").fillna(0),
            out["match_minutes"],
        )
    elif objective_damage_col:
        # NOTE: Fallback keeps objective pressure feature when turret damage is unavailable.
        out["objective_damage_per_min"] = _safe_ratio(
            out[objective_damage_col], out["match_minutes"]
        )

    if minions_col and neutral_col:
        # NOTE: CS/min tracks farming pace, including lane and jungle minions.
        out["cs_per_min"] = _safe_ratio(
            pd.to_numeric(out[minions_col], errors="coerce").fillna(0)
            + pd.to_numeric(out[neutral_col], errors="coerce").fillna(0),
            out["match_minutes"],
        )
    elif minions_col:
        # NOTE: Fallback keeps CS tempo using available lane-farm totals.
        out["cs_per_min"] = _safe_ratio(out[minions_col], out["match_minutes"])

    if vision_col:
        # NOTE: Vision per minute is a proxy for map information control.
        out["vision_per_min"] = _safe_ratio(out[vision_col], out["match_minutes"])

    if cc_col:
        # NOTE: CC per minute summarizes crowd-control contribution over game pace.
        out["cc_per_min"] = _safe_ratio(out[cc_col], out["match_minutes"])

    if own_jungle_col and enemy_jungle_col:
        # NOTE: Ratio uses own / (own + enemy) to normalize for game pace.
        out["jungle_control_share"] = _safe_ratio(
            out[own_jungle_col],
            pd.to_numeric(out[own_jungle_col], errors="coerce").fillna(0)
            + pd.to_numeric(out[enemy_jungle_col], errors="coerce").fillna(0),
        )

    return out
