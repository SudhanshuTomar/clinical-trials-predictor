import pandas as pd
import numpy as np
import pickle
from typing import Optional

def save_model(model, path: str):
    """
    Serialize a trained model to disk via pickle.

    Args:
        model: Trained model object.
        path: Filepath for .pkl file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str):
    """
    Load a pickled model from disk.

    Args:
        path: Filepath for .pkl file.

    Returns:
        Loaded model object.
    """
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m


def get_pts_scores(
    model,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Generate PTS% predictions and optional performance summary.

    Args:
        model: Trained classifier with predict_proba().
        X: Feature DataFrame.
        y: True labels (optional).
        threshold: Probability cutoff to compute metrics if y provided.

    Returns:
        DataFrame with columns ['pts%', 'predicted_class'] 
        and optionally 'recall', 'precision', 'auc' printed.
    """
    probs = model.predict_proba(X)[:, 1]
    pts = probs * 100
    df = pd.DataFrame({"pts%": pts})
    df["predicted_class"] = (probs > threshold).astype(int)

    if y is not None:
        from sklearn.metrics import recall_score, precision_score, roc_auc_score
        rec = recall_score(y, df["predicted_class"])
        prec = precision_score(y, df["predicted_class"])
        auc = roc_auc_score(y, probs)
        print(f"Recall: {rec:.3f}, Precision: {prec:.3f}, AUC: {auc:.3f}")

    return df


def export_predictions(
    df_preds: pd.DataFrame,
    id_series: pd.Series,
    path: str = "predictions.csv"
):
    """
    Export PTS predictions alongside trial IDs to CSV.

    Args:
        df_preds: DataFrame from get_pts_scores.
        id_series: Series of trial IDs aligned with df_preds.
        path: Output CSV filepath.
    """
    out = pd.concat([id_series.reset_index(drop=True), df_preds.reset_index(drop=True)], axis=1)
    out.columns = ["nct_number"] + list(df_preds.columns)
    out.to_csv(path, index=False)
    print(f"Predictions exported to {path}")
