import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE


def create_baseline_model(
    X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> Dict:
    """
    Train a simple logistic regression baseline.

    Args:
        X: Training features.
        y: Training labels (0/1).
        cv: Number of CV folds.

    Returns:
        Dict with keys:
          - 'model': the trained LogisticRegression
          - 'cv_mean': mean ROC AUC across folds
          - 'cv_std': std of ROC AUC across folds
          - 'cv_scores': array of fold scores
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return {"model": model, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}


def train_advanced_models(
    X: pd.DataFrame, y: pd.Series, cv: int = 5, n_iter: int = 10
) -> Dict[str, Dict]:
    """
    Train and tune tree‑based ensemble models (RF, XGB, GB).

    Args:
        X: Training features.
        y: Training labels.
        cv: CV folds.
        n_iter: RandomizedSearch iterations.

    Returns:
        Dict mapping model names to dicts with:
          - 'model': best estimator
          - 'cv_mean', 'cv_std'
          - 'feature_importance': DataFrame of top 10 features
    """
    def show_imp(m, name):
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
            df = pd.DataFrame({"feature": X.columns, "importance": imp})
            return df.nlargest(10, "importance")
        return None

    models = {}

    # RandomForest
    rf = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_cv = RandomizedSearchCV(
        rf, rf_grid, n_iter=n_iter, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
    )
    rf_cv.fit(X, y)
    best_rf = rf_cv.best_estimator_
    scores = cross_val_score(best_rf, X, y, cv=cv, scoring="roc_auc")
    models["RandomForest"] = {
        "model": best_rf,
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "feature_importance": show_imp(best_rf, "RF"),
    }

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }
    xgb_cv = RandomizedSearchCV(
        xgb_clf, xgb_grid, n_iter=n_iter, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
    )
    xgb_cv.fit(X, y)
    best_xgb = xgb_cv.best_estimator_
    scores = cross_val_score(best_xgb, X, y, cv=cv, scoring="roc_auc")
    models["XGBoost"] = {
        "model": best_xgb,
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "feature_importance": show_imp(best_xgb, "XGB"),
    }

    # GradientBoosting
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
    }
    gb_cv = RandomizedSearchCV(
        gb, gb_grid, n_iter=n_iter, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42
    )
    gb_cv.fit(X, y)
    best_gb = gb_cv.best_estimator_
    scores = cross_val_score(best_gb, X, y, cv=cv, scoring="roc_auc")
    models["GradientBoosting"] = {
        "model": best_gb,
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "feature_importance": show_imp(best_gb, "GB"),
    }

    return models


def create_ensemble_model(
    models: Dict[str, Dict], X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> Dict:
    """
    Build a soft-voting ensemble of top-3 models.

    Args:
        models: Output of train_advanced_models.
        X, y: Training data.
        cv: CV folds for ensemble.

    Returns:
        Dict with 'model', 'cv_mean', 'cv_std', 'component_models'
    """
    sorted_models = sorted(models.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
    top3 = sorted_models[:3]
    estimators = [(n, d["model"]) for n, d in top3]
    vc = VotingClassifier(estimators=estimators, voting="soft")
    vc.fit(X, y)
    scores = cross_val_score(vc, X, y, cv=cv, scoring="roc_auc")
    return {
        "model": vc,
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "component_models": [n for n, _ in top3],
    }


def calibrate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 3):
    """
    Calibrate classifier probabilities via isotonic regression.

    Args:
        model: Trained classifier.
        X, y: Data for calibration.
        cv: CV folds.

    Returns:
        CalibratedClassifierCV instance.
    """
    calib = CalibratedClassifierCV(model, method="isotonic", cv=cv)
    calib.fit(X, y)
    return calib
