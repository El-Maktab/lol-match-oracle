from __future__ import annotations

from typing import Any, Mapping, Protocol

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


def build_svm_model(
    model_name: str,
    *,
    params: Mapping[str, Any],
    random_state: int,
) -> SvmClassifier | None:
    """Factory entrypoint for SVM models.

    This module intentionally provides only interfaces in P4-A so P4-C can
    implement full SVM behavior without changing Trainer contracts.
    """

    del params
    del random_state

    normalized = model_name.strip().lower()
    if normalized not in SVM_MODEL_NAMES:
        return None

    # NOTE: P4-C will supply concrete SVM wrappers for this interface.
    raise NotImplementedError(
        f"Model '{normalized}' is reserved for P4-C implementation."
    )
