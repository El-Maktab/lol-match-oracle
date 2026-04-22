from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import optuna  # type: ignore[import-not-found]

TUNABLE_MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm_linear",
    "svm_rbf",
)

SPEC_FIELDS = ("kind", "low", "high", "step", "log", "choices")

DEFAULT_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "random_forest": {
        "n_estimators": {"kind": "int", "low": 100, "high": 800, "step": 50},
        "max_depth": {"kind": "int", "low": 4, "high": 20, "step": 1},
        "min_samples_split": {"kind": "int", "low": 2, "high": 10, "step": 1},
        "min_samples_leaf": {"kind": "int", "low": 1, "high": 5, "step": 1},
    },
    "xgboost": {
        "n_estimators": {"kind": "int", "low": 200, "high": 1200, "step": 50},
        "learning_rate": {"kind": "float", "low": 1e-3, "high": 0.3, "log": True},
        "max_depth": {"kind": "int", "low": 3, "high": 12, "step": 1},
        "subsample": {"kind": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"kind": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"kind": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"kind": "float", "low": 1e-8, "high": 10.0, "log": True},
        "min_child_weight": {"kind": "float", "low": 0.5, "high": 10.0, "log": True},
        "gamma": {"kind": "float", "low": 0.0, "high": 5.0},
    },
    "lightgbm": {
        "n_estimators": {"kind": "int", "low": 200, "high": 1200, "step": 50},
        "learning_rate": {"kind": "float", "low": 1e-3, "high": 0.3, "log": True},
        "num_leaves": {"kind": "int", "low": 15, "high": 127, "step": 1},
        "subsample": {"kind": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"kind": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"kind": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"kind": "float", "low": 1e-8, "high": 10.0, "log": True},
        "min_child_samples": {"kind": "int", "low": 5, "high": 100, "step": 1},
    },
    "svm_linear": {
        "c": {"kind": "float", "low": 1e-3, "high": 100.0, "log": True},
    },
    "svm_rbf": {
        "c": {"kind": "float", "low": 1e-3, "high": 100.0, "log": True},
        "gamma": {"kind": "float", "low": 1e-4, "high": 10.0, "log": True},
    },
}


@dataclass(slots=True)
class SearchSpaceResolution:
    """Resolved search space for one model after applying config overrides."""

    model_name: str
    param_specs: dict[str, dict[str, Any]]


def _parse_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("|", ",").split(",")]
        return [part for part in parts if part]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _parse_categorical_token(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _normalize_spec(param_name: str, raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(raw_spec.get("kind", "")).strip().lower()
    spec = dict(raw_spec)

    if not kind:
        if "choices" in spec:
            kind = "categorical"
        elif any(field in spec for field in ("low", "high")):
            low = spec.get("low")
            high = spec.get("high")
            if isinstance(low, int) and isinstance(high, int):
                kind = "int"
            else:
                kind = "float"
        else:
            raise ValueError(
                f"Search-space parameter '{param_name}' is missing 'kind'."
            )

    spec["kind"] = kind

    if kind in {"int", "float"}:
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"Search-space parameter '{param_name}' requires low/high bounds."
            )
        spec["log"] = bool(spec.get("log", False))

    if kind == "int":
        spec["low"] = int(spec["low"])
        spec["high"] = int(spec["high"])
        spec["step"] = int(spec.get("step", 1))

    if kind == "float":
        spec["low"] = float(spec["low"])
        spec["high"] = float(spec["high"])
        if "step" in spec and spec["step"] not in {None, ""}:
            spec["step"] = float(spec["step"])

    if kind == "categorical":
        choices_value = spec.get("choices")
        if choices_value is None:
            raise ValueError(
                f"Search-space parameter '{param_name}' requires categorical choices."
            )

        if isinstance(choices_value, str):
            parsed = _parse_csv(choices_value)
            choices = [_parse_categorical_token(token) for token in parsed]
        elif isinstance(choices_value, Sequence):
            choices = list(choices_value)
        else:
            choices = [choices_value]

        if not choices:
            raise ValueError(
                f"Search-space parameter '{param_name}' has empty categorical choices."
            )
        spec["choices"] = choices

    if kind not in {"int", "float", "categorical"}:
        raise ValueError(
            f"Unsupported search-space kind '{kind}' for parameter '{param_name}'."
        )

    return spec


def _extract_override_specs(
    model_name: str,
    config_mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prefix = f"optuna_{model_name}_"
    overrides: dict[str, dict[str, Any]] = {}

    for key, value in config_mapping.items():
        if not key.startswith(prefix):
            continue

        suffix = key[len(prefix) :]
        for field in SPEC_FIELDS:
            token = f"_{field}"
            if suffix.endswith(token):
                param_name = suffix[: -len(token)]
                if not param_name:
                    continue
                overrides.setdefault(param_name, {})[field] = value
                break

    return overrides


def resolve_search_space(
    model_name: str,
    config_mapping: Mapping[str, Any],
) -> SearchSpaceResolution:
    """Resolve model search-space by merging defaults with config overrides."""

    normalized = model_name.strip().lower()
    if normalized not in TUNABLE_MODEL_NAMES:
        raise ValueError(f"Unsupported model for tuning: '{normalized}'.")

    merged_specs = deepcopy(DEFAULT_SEARCH_SPACES[normalized])
    override_specs = _extract_override_specs(normalized, config_mapping)

    for param_name, override_spec in override_specs.items():
        if param_name in merged_specs:
            merged_specs[param_name].update(override_spec)
        else:
            merged_specs[param_name] = dict(override_spec)

    normalized_specs: dict[str, dict[str, Any]] = {}
    for param_name, raw_spec in merged_specs.items():
        normalized_specs[param_name] = _normalize_spec(param_name, raw_spec)

    return SearchSpaceResolution(
        model_name=normalized,
        param_specs=normalized_specs,
    )


def suggest_model_params(
    trial: optuna.Trial,
    *,
    model_name: str,
    config_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample hyperparameters for one trial using the resolved search-space."""

    resolution = resolve_search_space(model_name, config_mapping)

    params: dict[str, Any] = {}
    for param_name, spec in resolution.param_specs.items():
        kind = str(spec["kind"])

        if kind == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
            continue

        if kind == "float":
            if "step" in spec:
                params[param_name] = trial.suggest_float(
                    param_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    step=float(spec["step"]),
                    log=bool(spec.get("log", False)),
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            continue

        if kind == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name,
                list(spec["choices"]),
            )
            continue

        raise ValueError(f"Unsupported parameter kind '{kind}'.")

    return params


def get_configured_models(config_mapping: Mapping[str, Any]) -> tuple[str, ...]:
    """Return models to tune from config, defaulting to standard advanced set."""

    configured = _parse_csv(config_mapping.get("optuna_models"))
    if not configured:
        configured = ["random_forest", "xgboost", "lightgbm", "svm_rbf"]

    normalized = [item.strip().lower() for item in configured if item.strip()]

    unknown = [model for model in normalized if model not in TUNABLE_MODEL_NAMES]
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported model names in optuna_models: {joined}")

    return tuple(normalized)


def get_model_trial_budget(
    model_name: str,
    *,
    config_mapping: Mapping[str, Any],
    default_n_trials: int,
) -> int:
    """Read per-model trial budget override from config if available."""

    key = f"optuna_{model_name}_n_trials"
    raw_value = config_mapping.get(key)
    if raw_value is None:
        return int(default_n_trials)
    return max(1, int(raw_value))


__all__ = [
    "SearchSpaceResolution",
    "TUNABLE_MODEL_NAMES",
    "get_configured_models",
    "get_model_trial_budget",
    "resolve_search_space",
    "suggest_model_params",
]
