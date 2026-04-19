from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
WEBAPP_DIR = PROJECT_ROOT / "webapp"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_MIN_CURATED_DURATION_SECONDS = 300

BLUE_TEAM_ID = 100
RED_TEAM_ID = 200
TEAM_IDS = [BLUE_TEAM_ID, RED_TEAM_ID]
TEAM_KEY_COLUMNS = ["matchid", "teamid"]
EXPECTED_TEAM_ROWS_PER_MATCH = 2

TARGET_COLUMN = "win"

RAW_TABLE_FILES = {
    "champs": "champs.csv",
    "matches": "matches.csv",
    "participants": "participants.csv",
    "stats1": "stats1.csv",
    "stats2": "stats2.csv",
    "teambans": "teambans.csv",
    "teamstats": "teamstats.csv",
}

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

__all__ = [
    "BLUE_TEAM_ID",
    "BINARY_COLUMNS",
    "CONFIGS_DIR",
    "DATA_DIR",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_MIN_CURATED_DURATION_SECONDS",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VAL_SIZE",
    "DOCS_DIR",
    "EXPECTED_TEAM_ROWS_PER_MATCH",
    "EXTERNAL_DATA_DIR",
    "INTERIM_DATA_DIR",
    "MLRUNS_DIR",
    "MODELS_DIR",
    "NON_NEGATIVE_COLUMNS",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "RAW_TABLE_FILES",
    "RED_TEAM_ID",
    "TEAM_IDS",
    "TEAM_KEY_COLUMNS",
    "REPORTS_DIR",
    "SRC_DIR",
    "TARGET_COLUMN",
    "WEBAPP_DIR",
]
