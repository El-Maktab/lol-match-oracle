"""
Evaluation module for League of Legends Match Oracle.
"""

from .metrics import (
    calculate_classification_metrics,
    mcnemar_test,
)
from .report import export_evaluation_summary
from .visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_feature_importance,
)

__all__ = [
    "calculate_classification_metrics",
    "mcnemar_test",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_feature_importance",
    "export_evaluation_summary",
]
