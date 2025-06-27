# 🎯 Phase III Clinical Trial Success Predictor

This project predicts whether a Phase III clinical trial will succeed or fail, using **public metadata from ClinicalTrials.gov** — with no access to patient-level data or internal outcomes.

🧠 **Problem Statement**

> Can we estimate the outcome of a trial (Approved vs. Failed) using only public metadata: sponsor info, study design, intervention types, and operational details?

---

## 📦 Pipeline Overview
![Pipeline Flow Diagram](PipelineFlowDiagram.png)

---

## ⚙️ Key Features

### 🔍 Feature Engineering

* `study_design` → parsed into masking roles, intervention model, primary purpose
* `conditions` → mapped to therapeutic areas using keyword rules
* `interventions` → decomposed into types (drug, device, etc.), counts, presence flags
* Timeline metrics: duration, lag to results, etc.
* Sponsor approval rates computed from historical outcomes

### 📊 Modeling

* Baseline: Logistic Regression
* Tuned: Random Forest, XGBoost, GradientBoosting
* Ensemble: VotingClassifier (soft voting) + Isotonic Calibration
* Imbalance handled with SMOTE
* Final Model: Calibrated Ensemble
* AUC on validation: **0.86**

### 🎯 Outputs

* Probability of trial success (PTS %)
* Feature importance & SHAP explanations
* Complete pipeline with modular scripts for:

  * `data_ingest.py`
  * `feature_engineering.py`
  * `preprocessing.py`
  * `model_training.py`
  * `evaluation.py`
  * `shap_utils.py`
  * `main_pipeline.py`

---

## 🔬 SHAP Insights

* Most impactful: `sponsor_approval_rate`, `study_status`, `duration_days`
* Long & complex trials → higher risk of failure
* Industry-backed, focused trials → better chance of approval

---

## 📁 Folder Structure

```
├── data/                   # Raw and processed datasets
├── models/                 # Saved .pkl models
├── notebooks/              # Exploration and EDA
├── src/
│   ├── data_ingest.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── shap_utils.py
│   └── main_pipeline.py
└── README.md
```

---

## 💡 Run Pipeline

```bash
python src/main_pipeline.py
```

---

## 📌 Dependencies

Install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🧠 Author

Built and maintained by [Sudhanshu Tomar](https://www.linkedin.com/in/sudhanshu-tomar-12b49164/).
Mostly a weekend hackathon project — but one that turned into a full-fledged ML system. ⚙️
