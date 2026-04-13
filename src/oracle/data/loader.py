from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.constants import RAW_TABLE_FILES


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, nrows=nrows)


def load_stats_table(data_dir: str | Path, nrows: int | None = None) -> pd.DataFrame:
    data_path = Path(data_dir)
    stats1 = _read_csv(data_path / "stats1.csv", nrows=nrows)
    stats2 = _read_csv(data_path / "stats2.csv", nrows=nrows)

    # Keep a stable schema even if source column order changes.
    stats2 = stats2.reindex(columns=stats1.columns)
    stats = pd.concat([stats1, stats2], axis=0, ignore_index=True)
    return stats


def load_raw_tables(
    data_dir: str | Path,
    *,
    include_champs: bool = True,
    nrows: int | None = None,
) -> dict[str, pd.DataFrame]:
    data_path = Path(data_dir)

    tables: dict[str, pd.DataFrame] = {
        "matches": _read_csv(data_path / RAW_TABLE_FILES["matches"], nrows=nrows),
        "participants": _read_csv(
            data_path / RAW_TABLE_FILES["participants"], nrows=nrows
        ),
        "teamstats": _read_csv(data_path / RAW_TABLE_FILES["teamstats"], nrows=nrows),
        "teambans": _read_csv(data_path / RAW_TABLE_FILES["teambans"], nrows=nrows),
        "stats": load_stats_table(data_path, nrows=nrows),
    }

    if include_champs:
        tables["champs"] = _read_csv(data_path / RAW_TABLE_FILES["champs"], nrows=nrows)

    return tables
