from __future__ import annotations

import pandas as pd

from ..utils.constants import BLUE_TEAM_ID, RED_TEAM_ID, TEAM_IDS, TEAM_KEY_COLUMNS

_ROLE_ALIASES = {
    "top": "top",
    "solo": "top",
    "jungle": "jungle",
    "jungler": "jungle",
    "mid": "mid",
    "middle": "mid",
    "bot": "adc",
    "bottom": "adc",
    "adc": "adc",
    "duo_carry": "adc",
    "carry": "adc",
    "support": "support",
    "supp": "support",
    "utility": "support",
    "duo_support": "support",
}
_ROLE_ORDER = {
    "top": 0,
    "jungle": 1,
    "mid": 2,
    "adc": 3,
    "support": 4,
}
_SLOT_ROLE_FALLBACK = {
    1: "top",
    2: "jungle",
    3: "mid",
    4: "adc",
    5: "support",
}


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


def _team_player_slot(player_slot: object) -> int | None:
    if pd.isna(player_slot):
        return None
    slot = int(player_slot)
    return ((slot - 1) % 5) + 1


def _normalize_role(*values: object, player_slot: object) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if not text or text in {"none", "nan", "unknown"}:
            continue
        if text in _ROLE_ALIASES:
            return _ROLE_ALIASES[text]

    fallback_slot = _team_player_slot(player_slot)
    if fallback_slot is not None:
        return _SLOT_ROLE_FALLBACK.get(fallback_slot, "unknown")
    return "unknown"


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


def _pivot_slot_values(
    frame: pd.DataFrame,
    *,
    value_col: str,
    prefix: str,
    slot_col: str,
) -> pd.DataFrame:
    if frame.empty or value_col not in frame.columns:
        return pd.DataFrame(columns=TEAM_KEY_COLUMNS)

    pivoted = frame.pivot(
        index=TEAM_KEY_COLUMNS,
        columns=slot_col,
        values=value_col,
    ).reset_index()
    pivoted.columns = [
        col if isinstance(col, str) else f"{prefix}_{int(col)}"
        for col in pivoted.columns
    ]
    return pivoted


def _build_team_draft_rows(participant_team_df: pd.DataFrame) -> pd.DataFrame:
    draft_participants = participant_team_df.copy()
    draft_participants["canonical_role"] = draft_participants.apply(
        lambda row: _normalize_role(
            row.get("position"),
            row.get("role"),
            player_slot=row.get("player"),
        ),
        axis=1,
    )
    draft_participants["team_player_slot"] = draft_participants["player"].apply(
        _team_player_slot
    )
    draft_participants["role_rank"] = draft_participants["canonical_role"].map(
        _ROLE_ORDER
    )
    draft_participants["role_rank"] = draft_participants["role_rank"].fillna(99)

    ordered = draft_participants.sort_values(
        by=["matchid", "teamid", "role_rank", "team_player_slot", "player", "id"]
    ).copy()
    ordered["draft_slot"] = ordered.groupby(TEAM_KEY_COLUMNS).cumcount() + 1
    ordered = ordered.loc[ordered["draft_slot"] <= 5].copy()

    champion_slots = _pivot_slot_values(
        ordered,
        value_col="championid",
        prefix="champion",
        slot_col="draft_slot",
    )
    role_slots = _pivot_slot_values(
        ordered,
        value_col="canonical_role",
        prefix="role",
        slot_col="draft_slot",
    )

    role_specific = ordered.drop_duplicates(
        subset=TEAM_KEY_COLUMNS + ["canonical_role"], keep="first"
    )
    role_specific = role_specific.loc[
        role_specific["canonical_role"].isin(_ROLE_ORDER)
    ].copy()
    role_specific["role_feature"] = (
        role_specific["canonical_role"].astype("string") + "_championid"
    )
    role_pivot = role_specific.pivot(
        index=TEAM_KEY_COLUMNS,
        columns="role_feature",
        values="championid",
    ).reset_index()
    role_pivot.columns = [
        col if isinstance(col, str) else str(col) for col in role_pivot.columns
    ]

    return champion_slots.merge(role_slots, on=TEAM_KEY_COLUMNS, how="outer").merge(
        role_pivot,
        on=TEAM_KEY_COLUMNS,
        how="left",
    )


def _build_team_bans(teambans: pd.DataFrame) -> pd.DataFrame:
    bans = _coerce_int(teambans, ["matchid", "teamid", "championid", "banturn"])
    if not {"matchid", "teamid", "championid"}.issubset(bans.columns):
        return pd.DataFrame(columns=TEAM_KEY_COLUMNS + ["n_bans"])

    bans = bans.dropna(subset=TEAM_KEY_COLUMNS).copy()
    bans = bans.loc[bans["teamid"].isin(TEAM_IDS)].copy()
    if bans.empty:
        return pd.DataFrame(columns=TEAM_KEY_COLUMNS + ["n_bans"])

    if "banturn" in bans.columns:
        bans = bans.sort_values(by=["matchid", "teamid", "banturn", "championid"])
    else:
        bans = bans.sort_values(by=["matchid", "teamid", "championid"])

    bans["ban_slot"] = bans.groupby(TEAM_KEY_COLUMNS).cumcount() + 1
    bans = bans.loc[bans["ban_slot"] <= 5].copy()

    ban_slots = _pivot_slot_values(
        bans,
        value_col="championid",
        prefix="banned_champion",
        slot_col="ban_slot",
    )
    ban_counts = (
        bans.groupby(TEAM_KEY_COLUMNS, as_index=False)["championid"]
        .count()
        .rename(columns={"championid": "n_bans"})
    )
    return ban_counts.merge(ban_slots, on=TEAM_KEY_COLUMNS, how="left")


def merge_pregame_dataset(
    matches: pd.DataFrame,
    participants: pd.DataFrame,
    stats: pd.DataFrame,
    teambans: pd.DataFrame,
) -> pd.DataFrame:
    matches_ = _coerce_int(matches, ["id"])
    participants_ = _coerce_int(
        participants, ["id", "matchid", "player", "teamid", "championid"]
    )
    stats_ = _coerce_int(stats, ["id"])

    participant_merged = participants_.merge(
        stats_[[col for col in ["id", "win"] if col in stats_.columns]],
        on="id",
        how="inner",
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
    participant_merged = participant_merged.loc[
        participant_merged["teamid"].isin(TEAM_IDS)
    ].copy()

    team_win = _derive_team_win(participant_merged)
    team_draft = _build_team_draft_rows(participant_merged)
    team_bans = _build_team_bans(teambans)

    matches_by_team = matches_.rename(columns={"id": "matchid"})
    merged = team_win.merge(team_draft, on=TEAM_KEY_COLUMNS, how="left")
    merged = merged.merge(team_bans, on=TEAM_KEY_COLUMNS, how="left")
    merged = merged.merge(matches_by_team, on="matchid", how="left")
    merged["n_bans"] = pd.to_numeric(merged.get("n_bans", 0), errors="coerce").fillna(0)
    merged["is_blue_side"] = (merged["teamid"] == BLUE_TEAM_ID).astype("int8")

    return merged
