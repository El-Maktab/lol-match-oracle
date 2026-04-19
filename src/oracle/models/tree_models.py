from __future__ import annotations

from typing import Any, Mapping, Protocol

TREE_MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "lightgbm",
)


class TreeClassifier(Protocol):
    """Interface contract for tree-based classifiers used by the Trainer."""

    def fit(self, x: Any, y: Any) -> Any:
        """Fit model parameters on the provided training matrix."""

    def predict(self, x: Any) -> Any:
        """Predict class labels for the provided matrix."""


def build_tree_model(
    model_name: str,
    *,
    params: Mapping[str, Any],
    random_state: int,
) -> TreeClassifier | None:
    """Factory entrypoint for tree models.

    This module intentionally provides only interfaces in P4-A so P4-C can
    implement full tree-model behavior without changing Trainer contracts.
    """

    del params
    del random_state

    normalized = model_name.strip().lower()
    if normalized not in TREE_MODEL_NAMES:
        return None

    # NOTE: P4-C will supply concrete tree-model wrappers for this interface.
    raise NotImplementedError(
        f"Model '{normalized}' is reserved for P4-C implementation."
    )
