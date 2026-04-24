from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from sklearn.svm import SVC

SVM_MODEL_NAMES = (
    "svm_linear",
    "svm_rbf",
)


class SvmClassifier(Protocol):
    """Interface contract for SVM classifiers used by the Trainer."""

    def fit(self, x: Any, y: Any) -> Any:
        """Fit model parameters on the provided training matrix."""

    def predict(self, x: Any) -> Any:
        """Predict class labels for the provided matrix."""


@dataclass(slots=True)
class SupportVectorMachineClassifier:
    """SVC wrapper for linear and RBF kernels with trainer-compatible API."""

    kernel: str
    c: float
    gamma: str | float
    class_weight: str | dict[int, float] | None
    probability: bool
    max_iter: int
    max_train_samples: int | None
    random_state: int
    _model: SVC = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = SVC(
            kernel=self.kernel,
            C=self.c,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=self.probability,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> SupportVectorMachineClassifier:
        """Fit SVM model; validation split is accepted for API uniformity."""

        del x_val
        del y_val
        del progress_desc
        x_fit, y_fit = _downsample_training_set(
            x,
            y,
            max_train_samples=self.max_train_samples,
            random_state=self.random_state,
        )
        self._model.set_params(verbose=show_progress)
        self._model.fit(x_fit, y_fit)
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict binary class labels."""

        return np.asarray(self._model.predict(x), dtype=int)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities when enabled in the wrapped model."""

        if not self.probability:
            margins = self.decision_function(x)
            positive = 1.0 / (1.0 + np.exp(-margins))
            negative = 1.0 - positive
            return np.column_stack((negative, positive))

        return np.asarray(self._model.predict_proba(x), dtype=float)

    def decision_function(self, x: Any) -> np.ndarray:
        """Return signed margin distances used in ROC-AUC scoring."""

        return np.asarray(self._model.decision_function(x), dtype=float)


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

    raise TypeError("Unsupported class_weight value for SVM model.")


def _normalize_gamma(value: Any) -> str | float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"scale", "auto"}:
            return normalized

    return float(value)


def _downsample_training_set(
    x: Any,
    y: Any,
    *,
    max_train_samples: int | None,
    random_state: int,
) -> tuple[Any, Any]:
    if max_train_samples is None:
        return x, y

    cap = int(max_train_samples)
    if cap <= 0:
        return x, y

    total = len(y)
    if total <= cap:
        return x, y

    y_array = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(y_array, return_counts=True)
    if classes.size <= 1:
        selected = rng.choice(total, size=cap, replace=False)
    else:
        target_counts = np.maximum(
            1,
            np.floor((counts / counts.sum()) * cap).astype(int),
        )

        while target_counts.sum() > cap:
            largest = int(np.argmax(target_counts))
            if target_counts[largest] > 1:
                target_counts[largest] -= 1
            else:
                break

        while target_counts.sum() < cap:
            room = counts - target_counts
            target_counts[int(np.argmax(room))] += 1

        selected_parts: list[np.ndarray] = []
        for cls, target_count in zip(classes, target_counts, strict=False):
            class_indices = np.flatnonzero(y_array == cls)
            take = min(int(target_count), class_indices.size)
            if take > 0:
                selected_parts.append(
                    rng.choice(class_indices, size=take, replace=False)
                )

        selected = np.concatenate(selected_parts)
        if selected.size < cap:
            remaining = np.setdiff1d(np.arange(total), selected, assume_unique=False)
            extra = min(cap - selected.size, remaining.size)
            if extra > 0:
                selected = np.concatenate(
                    [selected, rng.choice(remaining, size=extra, replace=False)]
                )

    selected = np.sort(selected.astype(int))

    if hasattr(x, "iloc"):
        x_subset = x.iloc[selected]
    else:
        x_subset = np.asarray(x)[selected]

    if hasattr(y, "iloc"):
        y_subset = y.iloc[selected]
    else:
        y_subset = np.asarray(y)[selected]

    return x_subset, y_subset


def build_svm_model(
    model_name: str,
    *,
    params: Mapping[str, Any],
    random_state: int,
) -> SvmClassifier | None:
    """Build a supported SVM model with config-driven hyperparameters."""

    normalized = model_name.strip().lower()
    if normalized not in SVM_MODEL_NAMES:
        return None

    if normalized == "svm_linear":
        return SupportVectorMachineClassifier(
            kernel="linear",
            c=float(params.get("c", 1.0)),
            gamma="scale",
            class_weight=_normalize_class_weight(params.get("class_weight", None)),
            probability=bool(params.get("probability", False)),
            max_iter=int(params.get("max_iter", 3000)),
            max_train_samples=(
                int(params.get("max_train_samples"))
                if params.get("max_train_samples") is not None
                else 60_000
            ),
            random_state=random_state,
        )

    if normalized == "svm_rbf":
        return SupportVectorMachineClassifier(
            kernel="rbf",
            c=float(params.get("c", 1.0)),
            gamma=_normalize_gamma(params.get("gamma", "scale")),
            class_weight=_normalize_class_weight(params.get("class_weight", None)),
            probability=bool(params.get("probability", False)),
            max_iter=int(params.get("max_iter", 3000)),
            max_train_samples=(
                int(params.get("max_train_samples"))
                if params.get("max_train_samples") is not None
                else 30_000
            ),
            random_state=random_state,
        )

    raise ValueError(f"Unsupported SVM model '{normalized}'.")
