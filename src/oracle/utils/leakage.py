from __future__ import annotations

from typing import Iterable

TARGET_LINKED_EXACT = {
    "participant_win_consistency",
    "participant_win_inconsistent",
    "champion_signal_prior",
    "assists_sum_diff_vs_opp",
    "dragonkills_diff_vs_opp",
    "inhibkills_diff_vs_opp",
}

TARGET_LINKED_SUBSTRINGS = (
    "participant_win",
    "turrent"
    "_winrate_te",
    "target_encoding",
    "label_encoding",
)

# NOTE: First-objective flags are highly target-proximal and can inflate holdout metrics.
OUTCOME_PROXY_EXACT = {
    "firstblood",
    "firstbloodkill",
    "firsttower",
    "firstinhib",
    "firstbaron",
    "firstdragon",
    "firstharry",
}


def is_leaky_feature_column(column: str, *, target_col: str) -> bool:
    """Return True when a feature is target-derived or outcome-proximal."""

    normalized = column.strip().lower()
    target_normalized = target_col.strip().lower()

    if normalized in TARGET_LINKED_EXACT or normalized in OUTCOME_PROXY_EXACT:
        return True

    if any(token in normalized for token in TARGET_LINKED_SUBSTRINGS):
        return True

    if (
        normalized.startswith("target_")
        or normalized.endswith("_target")
        or normalized.startswith("label_")
        or normalized.endswith("_label")
    ):
        return True

    if (
        normalized == target_normalized
        or normalized.startswith(f"{target_normalized}_")
        or normalized.endswith(f"_{target_normalized}")
        or f"_{target_normalized}_" in normalized
    ):
        return True

    return False


def split_leaky_feature_columns(
    feature_columns: Iterable[str], *, target_col: str
) -> tuple[list[str], list[str]]:
    """Split a feature list into safe and dropped(leaky) columns."""

    safe: list[str] = []
    dropped: list[str] = []

    for column in feature_columns:
        if is_leaky_feature_column(column, target_col=target_col):
            dropped.append(column)
        else:
            safe.append(column)

    return safe, dropped
