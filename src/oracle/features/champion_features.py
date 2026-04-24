"""Champion-based target encoding features for team-level modeling.

This module learns smoothed win-rate encodings from training data and applies
them to train/validation/test splits.

Main outputs:
- <source_column>_winrate_te:
    Smoothed historical win-rate signal for each champion-like categorical column.
- champion_signal_prior:
    Global fallback win-rate when no champion columns are available.

The objective is to convert sparse champion IDs into stable numeric priors while
reducing noise from low-frequency picks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ChampionEncodingArtifacts:
    """Stores fitted target-encoding mappings and global fallback rate."""

    mappings: dict[str, pd.Series]
    global_rate: float


def _fit_target_encoding(
    frame: pd.DataFrame,
    *,
    feature: str,
    target_col: str,
    smoothing: float,
) -> pd.Series:
    grouped = frame.groupby(feature, dropna=True)[target_col].agg(["mean", "count"])
    prior = float(frame[target_col].mean()) if len(frame) else 0.5
    # NOTE: We use smoothed target encoding to reduce variance for low-frequency champions.
    return (grouped["mean"] * grouped["count"] + prior * smoothing) / (
        grouped["count"] + smoothing
    )


def fit_champion_encoders(
    train_df: pd.DataFrame,
    *,
    target_col: str = "win",
    smoothing: float = 25.0,
) -> ChampionEncodingArtifacts:
    """Fit champion-like target encodings when champion columns are available."""
    # NOTE: Candidate columns cover both raw participant schema and team-aggregated schema.
    candidate_columns = [
        "championid",
        "championid_mode",
        "champion_1",
        "champion_2",
        "champion_3",
        "champion_4",
        "champion_5",
        "top_championid",
        "jungle_championid",
        "mid_championid",
        "adc_championid",
        "support_championid",
        "banned_champion_1",
        "banned_champion_2",
        "banned_champion_3",
        "banned_champion_4",
        "banned_champion_5",
    ]
    present = [col for col in candidate_columns if col in train_df.columns]

    mappings: dict[str, pd.Series] = {}
    for col in present:
        if train_df[col].isna().all():
            continue
        mappings[col] = _fit_target_encoding(
            train_df,
            feature=col,
            target_col=target_col,
            smoothing=smoothing,
        )

    # NOTE: global_rate is the default prior used for unseen or missing categories.
    return ChampionEncodingArtifacts(
        mappings=mappings,
        global_rate=float(train_df[target_col].mean()) if len(train_df) else 0.5,
    )


def apply_champion_encoders(
    frame: pd.DataFrame,
    artifacts: ChampionEncodingArtifacts,
) -> pd.DataFrame:
    out = frame.copy()

    for source_col, mapping in artifacts.mappings.items():
        encoded_col = f"{source_col}_winrate_te"
        # NOTE: Each *_winrate_te feature encodes champion identity as a smoothed win prior.
        out[encoded_col] = out[source_col].map(mapping).fillna(artifacts.global_rate)

    if not artifacts.mappings:
        # NOTE: Fallback keeps one weak champion signal when no encoding source exists.
        out["champion_signal_prior"] = artifacts.global_rate

    return out
