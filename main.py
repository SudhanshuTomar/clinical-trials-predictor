#!/usr/bin/env python3
import os
import pandas as pd
from src.data_ingest import load_nct_ids_from_csv, fetch_trials_data
from src.feature_engineer import engineering_pipeline, clean_column_names
from src.data_prep import prepare_datasets
from src.modeling import (
    create_baseline_model,
    train_advanced_models,
    create_ensemble_model,
    calibrate_model,
)
from src.evaluation import evaluate_model, plot_pr_roc_curves, plot_calibration_curve
from src.prediction import save_model, get_pts_scores, export_predictions
import os
from datetime import datetime

# 1. Configuration
TRAIN_IDS_CSV = "data/raw/Train.csv"
ACTIVE_IDS_CSV = "data/raw/Test.csv"
HIST_DATA_CSV = "data/raw/historical_trials_data.csv"
ACTIVE_DATA_CSV = "data/raw/active_trials_data.csv"
FEATURE_CSV     = "data/processed/feat_df_final.csv"
today_str      = datetime.now().strftime("%Y%m%d")
MODEL_PKL      = f"models/final_model_{today_str}.pkl"
PREDICTIONS_CSV= f"predictions/active_trial_pts_{today_str}.csv"

# 2. Step 1: Data Ingestion
if not os.path.exists(HIST_DATA_CSV):
    train_ids = load_nct_ids_from_csv(TRAIN_IDS_CSV, id_column="NCT Number")
    fetch_trials_data(train_ids, HIST_DATA_CSV)

if not os.path.exists(ACTIVE_DATA_CSV):
    active_ids = load_nct_ids_from_csv(ACTIVE_IDS_CSV, id_column="Trial_ID")
    fetch_trials_data(active_ids, ACTIVE_DATA_CSV)

#Merge outcome column from train_ids_csv into hist_data_csv
df_train = pd.read_csv(TRAIN_IDS_CSV)
df_hist = pd.read_csv(HIST_DATA_CSV)
df_hist = df_hist.merge(df_train[['NCT Number', 'Outcome']], on='NCT Number', how='left')
df_hist.to_csv(HIST_DATA_CSV, index=False)

# 3. Step 2: Feature Engineering
if not os.path.exists(FEATURE_CSV):
    df_hist   = clean_column_names(__import__("pandas").read_csv(HIST_DATA_CSV))
    df_active = clean_column_names(__import__("pandas").read_csv(ACTIVE_DATA_CSV))
    # Combine if you want to compute historical rates jointly
    df_all = df_hist.append(df_active, ignore_index=True)
    df_feat = engineering_pipeline(df_all)
    df_feat.to_csv(FEATURE_CSV, index=False)
else:
    df_feat = clean_column_names(__import__("pandas").read_csv(FEATURE_CSV))

# 4. Step 3: Pre‑Modeling Preparation
input_columns = [  # same list defined in your pipeline
    # ... paste your input_cols here ...
]
datasets = prepare_datasets(FEATURE_CSV, cleaner=clean_column_names, input_cols=input_columns)
X_train, y_train = datasets["X_train"], datasets["y_train"]
X_val,   y_val   = datasets["X_val"],   datasets["y_val"]
X_test,  y_test  = datasets["X_test"],  datasets["y_test"]

# 5. Step 4: Modeling
# 5.1 Baseline
baseline_res = create_baseline_model(X_train, y_train)
print("Baseline ROC AUC:", baseline_res["cv_mean"])

# 5.2 Advanced
adv_models = train_advanced_models(X_train, y_train)

# 5.3 Ensemble + Calibration
ensemble_res = create_ensemble_model(adv_models, X_train, y_train)
calibrated = calibrate_model(ensemble_res["model"], X_train, y_train)

# 6. Step 5: Evaluation on Validation Set
eval_res = evaluate_model(
    calibrated, 
    X_val, 
    y_val, 
    group_df=__import__("pandas").read_csv(ACTIVE_DATA_CSV), 
    group_col="therapeutic_area"
)
print("Validation AUC:", eval_res["auc"])
print("Per-TA AUC:\n", eval_res.get("per_group"))

# 7. Step 6: Final Test Evaluation & Calibration Plot
eval_test = evaluate_model(calibrated, X_test, y_test)
plot_pr_roc_curves(y_test.values, eval_test["y_proba"], "(Test)")
plot_calibration_curve(y_test.values, eval_test["y_proba"])

# 8. Step 7: Save Final Model
save_model(calibrated, MODEL_PKL)

# 9. Step 8: Generate PTS Scores for Active Trials
#    Assume X_test corresponds to active set in same order as ACTIVE_DATA_CSV
pts_df = get_pts_scores(calibrated, X_test, y=None)
export_predictions(pts_df, __import__("pandas").read_csv(ACTIVE_DATA_CSV)["nct_number"], PREDICTIONS_CSV)

print("🚀 Pipeline complete. Predictions written to:", PREDICTIONS_CSV)
