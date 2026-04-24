from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron as SklearnPerceptron

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


@dataclass(slots=True)
class PerceptronClassifier:
    """Deterministic wrapper around sklearn's Perceptron baseline."""

    max_iter: int
    eta0: float
    penalty: str | None
    alpha: float
    tol: float
    fit_intercept: bool
    class_weight: str | dict[int, float] | None
    random_state: int
    _model: SklearnPerceptron = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = SklearnPerceptron(
            max_iter=self.max_iter,
            eta0=self.eta0,
            penalty=self.penalty,
            alpha=self.alpha,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, x: Any, y: Any) -> PerceptronClassifier:
        """Fit perceptron parameters on the training split."""

        self._model.fit(x, y)
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict binary class labels."""

        return self._model.predict(x).astype(int)

    def decision_function(self, x: Any) -> np.ndarray:
        """Return signed distances used by Trainer ROC-AUC scoring."""

        return np.asarray(self._model.decision_function(x), dtype=float)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Approximate probabilities from perceptron margins via sigmoid."""

        margins = self.decision_function(x)
        positive = 1.0 / (1.0 + np.exp(-margins))
        negative = 1.0 - positive
        return np.column_stack((negative, positive))


@dataclass(slots=True)
class LinearRegressionClassifier:
    """Closed-form linear-regression classifier using pseudo-inverse."""

    fit_intercept: bool
    ridge_alpha: float
    decision_threshold: float
    coef_: np.ndarray | None = field(init=False, default=None)
    intercept_: float = field(init=False, default=0.0)
    classes_: np.ndarray = field(
        init=False,
        default_factory=lambda: np.array([0, 1], dtype=int),
    )

    def __post_init__(self) -> None:
        self.coef_ = None
        self.intercept_ = 0.0
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, x: Any, y: Any) -> LinearRegressionClassifier:
        """Solve linear weights with pseudo-inverse on the train split only."""

        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        self._validate_targets(y_array)

        design = self._with_intercept(x_array)
        self._validate_ridge_alpha()

        if self.ridge_alpha > 0.0:
            # NOTE: Closed-form ridge stabilizes inversion on collinear features.
            regularizer = self.ridge_alpha * np.eye(design.shape[1], dtype=float)
            if self.fit_intercept:
                regularizer[0, 0] = 0.0
            gram = design.T @ design + regularizer
            weights = np.linalg.solve(gram, design.T @ y_array)
        else:
            weights = np.linalg.pinv(design) @ y_array

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = np.asarray(weights[1:], dtype=float)
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(weights, dtype=float)

        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict binary labels from linear-regression scores."""

        probabilities = self.predict_proba(x)[:, 1]
        return (probabilities >= self.decision_threshold).astype(int)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Map linear scores to [0, 1] via logistic link for metric parity."""

        scores = self.decision_function(x)
        positive = 1.0 / (1.0 + np.exp(-scores))
        negative = 1.0 - positive
        return np.column_stack((negative, positive))

    def decision_function(self, x: Any) -> np.ndarray:
        """Return raw linear scores before thresholding."""

        self._ensure_fitted()
        x_array = np.asarray(x, dtype=float)
        return np.asarray(x_array @ self.coef_ + self.intercept_, dtype=float)

    def _ensure_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

    def _with_intercept(self, x_array: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return x_array

        ones = np.ones((x_array.shape[0], 1), dtype=float)
        return np.hstack((ones, x_array))

    def _validate_targets(self, y_array: np.ndarray) -> None:
        unique = set(np.unique(y_array).tolist())
        if unique - {0.0, 1.0}:
            raise ValueError(
                "LinearRegressionClassifier expects binary targets in {0, 1}."
            )

    def _validate_ridge_alpha(self) -> None:
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")


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


def _normalize_penalty(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"none", "null", "~", ""}:
            return None
        return normalized

    return str(value)


def _build_logistic_regression(
    params: Mapping[str, Any],
    *,
    random_state: int,
) -> LogisticRegression:
    """Build a deterministic logistic regression baseline."""

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


def _build_perceptron(
    params: Mapping[str, Any],
    *,
    random_state: int,
) -> PerceptronClassifier:
    """Build deterministic PLA-style perceptron baseline."""

    return PerceptronClassifier(
        max_iter=int(params.get("max_iter", 1000)),
        eta0=float(params.get("eta0", 1.0)),
        penalty=_normalize_penalty(params.get("penalty", None)),
        alpha=float(params.get("alpha", 0.0001)),
        tol=float(params.get("tol", 1e-3)),
        fit_intercept=bool(params.get("fit_intercept", True)),
        class_weight=_normalize_class_weight(params.get("class_weight", None)),
        random_state=random_state,
    )


def _build_linear_regression_classifier(
    params: Mapping[str, Any],
) -> LinearRegressionClassifier:
    """Build closed-form linear-regression classifier baseline."""

    return LinearRegressionClassifier(
        fit_intercept=bool(params.get("fit_intercept", True)),
        ridge_alpha=float(params.get("ridge_alpha", 0.0)),
        decision_threshold=float(params.get("decision_threshold", 0.5)),
    )


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

    if normalized == "perceptron":
        return _build_perceptron(params, random_state=random_state)

    if normalized == "linear_regression_classifier":
        return _build_linear_regression_classifier(params)

    raise ValueError(f"Unsupported baseline model '{normalized}'.")
