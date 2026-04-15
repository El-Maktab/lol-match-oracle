from __future__ import annotations

import pandas as pd

BINARY_COLUMNS = [
    "win",
    "firstblood",
    "firsttower",
    "firstinhib",
    "firstbaron",
    "firstdragon",
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

    return cleaned
