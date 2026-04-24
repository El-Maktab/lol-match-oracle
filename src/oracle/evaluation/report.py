from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def export_evaluation_summary(
    metrics: Mapping[str, Any],
    best_model_name: str,
    output_dir: str | Path,
    file_name: str = "evaluation_summary.json",
) -> Path:
    """Write a normalized evaluation summary artifact for the selected champion."""

    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_metrics = _to_builtin(metrics)
    champion_metrics = normalized_metrics.get(best_model_name, {})

    report = {
        "champion_model": best_model_name,
        "champion_metrics": champion_metrics,
        "evaluation_metrics": normalized_metrics,
        "note": (
            "Champion model selected from held-out evaluation results with explicit "
            "generalization and bias-variance review."
        ),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return output_path
