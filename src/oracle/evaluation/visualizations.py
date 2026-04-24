import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.calibration import CalibrationDisplay

def plot_confusion_matrix(y_true, y_pred, ax=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_xticklabels(['Loss (0)', 'Win (1)'])
    ax.set_yticklabels(['Loss (0)', 'Win (1)'])
    return ax

def plot_roc_curve(models_dict, y_true, ax=None, title="ROC Curve"):
    """
    Plot ROC curve for multiple models.
    
    Args:
        models_dict: dict of {model_name: y_prob}
        y_true: True labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    for name, y_prob in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax

def plot_pr_curve(models_dict, y_true, ax=None, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve for multiple models.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    for name, y_prob in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')
        
    # Baseline is proportion of positive examples
    baseline = np.mean(y_true)
    ax.axhline(baseline, color='gray', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    return ax

def plot_calibration_curve(models_dict, y_true, ax=None, title="Calibration Curve (Reliability Diagram)"):
    """
    Plot calibration curve for multiple models.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    for name, y_prob in models_dict.items():
        CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, name=name, ax=ax)
        
    ax.set_title(title)
    return ax

def plot_feature_importance(importances, feature_names, top_n=20, ax=None, title="Feature Importance"):
    """
    Plot top feature importances.
    
    Args:
        importances: array of importance values
        feature_names: list of feature names
        top_n: number of top features to plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Create dataframe and sort
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    df = df.sort_values('Importance', ascending=False).head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=df, ax=ax, palette='viridis')
    ax.set_title(title)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    
    return ax
