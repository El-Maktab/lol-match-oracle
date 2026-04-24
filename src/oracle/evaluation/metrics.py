from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
) -> dict[str, float]:
    """Calculate stable classification metrics for binary evaluation."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
    }

    if y_prob is None:
        return metrics

    y_prob_array = np.asarray(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob_array, 1e-15, 1 - 1e-15)

    if np.unique(y_true_array).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob_clipped))
        metrics["pr_auc"] = float(average_precision_score(y_true_array, y_prob_clipped))

    metrics["brier_score"] = float(brier_score_loss(y_true_array, y_prob_clipped))
    metrics["log_loss"] = float(log_loss(y_true_array, y_prob_clipped))
    return metrics


def mcnemar_test(y_true: Any, y_pred1: Any, y_pred2: Any) -> dict[str, Any]:
    """Perform McNemar's test for two binary classifiers on the same labels."""

    y_true_array = np.asarray(y_true)
    y_pred1_array = np.asarray(y_pred1)
    y_pred2_array = np.asarray(y_pred2)

    if not (y_true_array.shape == y_pred1_array.shape == y_pred2_array.shape):
        raise ValueError("McNemar inputs must have identical shapes.")

    both_correct = int(
        np.sum((y_pred1_array == y_true_array) & (y_pred2_array == y_true_array))
    )
    first_only = int(
        np.sum((y_pred1_array == y_true_array) & (y_pred2_array != y_true_array))
    )
    second_only = int(
        np.sum((y_pred1_array != y_true_array) & (y_pred2_array == y_true_array))
    )
    both_wrong = int(
        np.sum((y_pred1_array != y_true_array) & (y_pred2_array != y_true_array))
    )

    table = [[both_correct, first_only], [second_only, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)

    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "table": table,
    }
