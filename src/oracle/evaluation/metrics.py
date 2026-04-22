import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    log_loss,
)
from statsmodels.stats.contingency_tables import mcnemar

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (optional)
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
        metrics["log_loss"] = log_loss(y_true, y_prob)
        
    return metrics

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test to compare two models.
    
    Args:
        y_true: True labels
        y_pred1: Predicted labels from model 1
        y_pred2: Predicted labels from model 2
        
    Returns:
        dict: Test statistic and p-value
    """
    # Create contingency table
    # Both correct
    n00 = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    # 1 correct, 2 incorrect
    n01 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    # 1 incorrect, 2 correct
    n10 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    # Both incorrect
    n11 = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    table = [[n00, n01], [n10, n11]]
    
    result = mcnemar(table, exact=False, correction=True)
    
    return {
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "table": table
    }
