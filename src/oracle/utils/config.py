from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    CONFIGS_DIR,
    DATA_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


def _parse_scalar(value: str) -> Any:
    text = value.strip()

    if not text:
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none", "~"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    loaded: dict[str, Any] = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                raise ValueError(
                    f"Unsupported YAML line in {config_path}: {raw_line.rstrip()}"
                )

            key, raw_value = line.split(":", 1)
            loaded[key.strip()] = _parse_scalar(raw_value)

    return loaded


@dataclass(slots=True)
class DataConfig:
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DATA_DIR
    interim_dir: Path = INTERIM_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    target_column: str = TARGET_COLUMN
    group_column: str | None = "matchid"
    test_size: float = DEFAULT_TEST_SIZE
    val_size: float = DEFAULT_VAL_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    include_champs: bool = True

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "DataConfig":
        base_dir = base_dir or CONFIGS_DIR.parent

        return cls(
            data_dir=_resolve_path(mapping.get("data_dir", DATA_DIR), base_dir),
            raw_dir=_resolve_path(mapping.get("raw_dir", RAW_DATA_DIR), base_dir),
            interim_dir=_resolve_path(
                mapping.get("interim_dir", INTERIM_DATA_DIR), base_dir
            ),
            processed_dir=_resolve_path(
                mapping.get("processed_dir", PROCESSED_DATA_DIR), base_dir
            ),
            target_column=str(mapping.get("target_column", TARGET_COLUMN)),
            group_column=(
                None
                if mapping.get("group_column", "matchid") is None
                else str(mapping.get("group_column", "matchid"))
            ),
            test_size=float(mapping.get("test_size", DEFAULT_TEST_SIZE)),
            val_size=float(mapping.get("val_size", DEFAULT_VAL_SIZE)),
            random_state=int(mapping.get("random_state", DEFAULT_RANDOM_STATE)),
            include_champs=bool(mapping.get("include_champs", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "raw_dir": self.raw_dir,
            "interim_dir": self.interim_dir,
            "processed_dir": self.processed_dir,
            "target_column": self.target_column,
            "group_column": self.group_column,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "include_champs": self.include_champs,
        }


def load_data_config(path: str | Path | None = None) -> DataConfig:
    config_path = Path(path) if path is not None else CONFIGS_DIR / "data.yaml"
    return DataConfig.from_mapping(
        load_yaml_config(config_path),
        base_dir=CONFIGS_DIR.parent,
    )
