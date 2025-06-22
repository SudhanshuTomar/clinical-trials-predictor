import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    calibration_curve
)

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    group_df: pd.DataFrame = None,
    group_col: str = "therapeutic_area",
) -> Dict:
    """
    Evaluate a classifier on the test set and compute a variety of metrics.

    Args:
        model: Fitted classifier with predict_proba().
        X_test: Features for test set.
        y_test: True binary labels.
        group_df: (Optional) DataFrame aligned with X_test/y_test for per-group metrics.
        group_col: Column name in group_df to group by (e.g. 'therapeutic_area').

    Returns:
        Dict containing:
          - 'auc': overall ROC AUC
          - 'avg_precision': overall average precision
          - 'brier_score': overall Brier score
          - 'classification_report': sklearn report dict
          - 'confusion_matrix': array
          - 'y_proba': raw positive-class probabilities
          - 'per_group': DataFrame of metrics by group (if group_df provided)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Overall metrics
    auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "auc": auc,
        "avg_precision": avg_prec,
        "brier_score": brier,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_proba": y_proba,
    }

    # Per-group analysis
    if group_df is not None and group_col in group_df:
        df = pd.DataFrame({
            "y_true": y_test.values,
            "y_proba": y_proba,
            group_col: group_df[group_col].values
        })
        metrics = []
        for area, sub in df.groupby(group_col):
            try:
                area_auc = roc_auc_score(sub["y_true"], sub["y_proba"])
            except ValueError:
                area_auc = np.nan
            metrics.append({"group": area, "auc": area_auc, "n": len(sub)})
        results["per_group"] = pd.DataFrame(metrics).sort_values("auc", ascending=False)

    return results


def plot_pr_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, title_suffix: str = ""):
    """
    Plots Precision-Recall and ROC curves.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        title_suffix: Optional suffix for the plot titles.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall
    axes[0].plot(recall, precision, lw=2)
    axes[0].fill_between(recall, precision, alpha=0.2)
    axes[0].set_title(f"Precision-Recall {title_suffix}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].text(0.6, 0.4, f"AP={avg_prec:.2f}")

    # ROC
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC={auc:.2f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title(f"ROC Curve {title_suffix}")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve"
):
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        n_bins: Number of bins to use.
        title: Plot title.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "s-", label="Empirical")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(f"{title}\nBrier Score = {brier:.3f}")
    plt.legend()
    plt.grid(True)
    plt.show()
