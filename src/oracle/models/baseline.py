from __future__ import annotations

from typing import Any, Mapping, Protocol

from sklearn.linear_model import LogisticRegression

BASELINE_MODEL_NAMES = (
    "logistic_regression",
    "perceptron",
    "linear_regression_classifier",
)


class TrainableClassifier(Protocol):
    """Interface contract for classifier wrappers used by the Trainer."""

    def fit(self, x: Any, y: Any) -> Any:
        """Fit model parameters on the provided training matrix."""

    def predict(self, x: Any) -> Any:
        """Predict class labels for the provided matrix."""


def _normalize_class_weight(value: Any) -> str | dict[int, float] | None:
    if value is None:
        return None

    if isinstance(value, str) and value.lower() in {"none", "null", "~", ""}:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        normalized: dict[int, float] = {}
        for key, weight in value.items():
            normalized[int(key)] = float(weight)
        return normalized

    raise TypeError("Unsupported class_weight value for logistic regression baseline.")


def _build_logistic_regression(
    params: Mapping[str, Any],
    *,
    random_state: int,
) -> LogisticRegression:
    """Build a deterministic logistic regression baseline.
    """

    # NOTE: sklearn deprecates explicit penalty='l2'; default behavior already applies L2.
    penalty = str(params.get("penalty", "l2")).lower()
    logistic_kwargs: dict[str, Any] = {
        "C": float(params.get("c", 1.0)),
        "solver": str(params.get("solver", "lbfgs")),
        "max_iter": int(params.get("max_iter", 1000)),
        "class_weight": _normalize_class_weight(params.get("class_weight", "balanced")),
        "random_state": random_state,
    }
    if penalty not in {"l2", "none", "null", "~", ""}:
        logistic_kwargs["penalty"] = penalty

    return LogisticRegression(**logistic_kwargs)


def build_baseline_model(
    model_name: str,
    *,
    params: Mapping[str, Any],
    random_state: int,
) -> TrainableClassifier | None:
    """Build a baseline model if supported, otherwise return None.

    Args:
        model_name: Canonical model key.
        params: Model-specific hyperparameters.
        random_state: Seed for deterministic behavior.

    Returns:
        A trainable classifier implementation or None if the model key does not
        belong to the baseline family.
    """

    normalized = model_name.strip().lower()
    if normalized not in BASELINE_MODEL_NAMES:
        return None

    if normalized == "logistic_regression":
        return _build_logistic_regression(params, random_state=random_state)

    # NOTE: P4-B owns the full baseline suite (Perceptron and linear-regression classifier).
    raise NotImplementedError(
        f"Model '{normalized}' is reserved for P4-B implementation. "
        "Use 'logistic_regression' for P4-A baseline runs."
    )
