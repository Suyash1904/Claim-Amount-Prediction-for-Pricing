# Auto Insurance Claims Cost Modeling

This repository contains an end-to-end machine learning pipeline to predict auto insurance claim costs (`total_claim_amount`) at the moment a claim is filed. It handles data preprocessing, feature engineering, strict target leakage prevention, and hyperparameter tuning across several gradient boosting algorithms.

---

## Table of Contents
* [Overview](#overview)
* [Data Quality Notes](#data-quality-notes)
* [The Target Leakage Problem](#the-target-leakage-problem)
* [Feature Engineering](#feature-engineering)
* [Model Benchmark](#model-benchmark)
* [Project Structure](#project-structure)
* [How to Run](#how-to-run)

---

## Overview
Predicting the severity (cost) of an insurance claim at the time it is filed is a core problem in actuarial science and claims management. Accurate cost prediction enables insurance providers to:
1. Optimize pricing structures and adjust policy premiums dynamically.
2. Flag high-cost claims early for senior adjuster routing.
3. Fast-track low-cost, low-risk claims for automatic settlement, reducing administrative overhead.

The best performing model (CatBoost) explains roughly 70% of the variance in claim amounts on unseen test data, achieving a Test RMSE of ~$14k and a Test MAE of ~$10k.
---
## Data Quality Notes
During Exploratory Data Analysis (EDA), we identified a critical data anomaly:
* **Negative Policy Tenure (Policy #794731)**: The claim incident occurred on **2015-02-02**, but the policy was bound on **2015-02-22**, resulting in a tenure of **-20 days**.
* **Implication**: This represents potential policy backdating, a system entry error, or high-risk fraud. We flagged this for risk management and kept the data to let the model learn the anomaly.

---

## The Target Leakage Problem
A major challenge in claims cost modeling is target leakage. The dataset contains sub-claim breakdowns:
* `injury_claim`
* `property_claim`
* `vehicle_claim`

These three columns sum exactly to the target `total_claim_amount`. If included in the training data, any model will achieve a trivial $R^2 = 1.0$. However, this is textbook data leakage because sub-claims are settled and calculated *after* the claim has been closed. 

To ensure this model actually works in production at the time of claim registration, we strictly dropped these features. We also dropped `fraud_reported` since fraud investigations conclude post-incident.

---

## Feature Engineering
To capture risk exposure, we engineered:
* **`policy_tenure_at_incident`**: Days between the policy bind date and the incident date.
* **`vehicle_age_at_incident`**: Years between the incident year and the vehicle's manufacturing year (`auto_year`). Older vehicles are more likely to be declared a total loss at lower caps, while new cars generate higher claim potential.
* **Temporal Attributes**: `incident_month` and `incident_day_of_week` to capture basic seasonality.

---

## Model Benchmark

We evaluated standard algorithms against advanced gradient boosting engines with native categorical support (using a log-transformed target).

| Model | Target Scale | Train RMSE ($) | Test RMSE ($) | Test MAE ($) | Test $R^2$ | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | Raw | 13,316.62 | 15,298.23 | 11,641.45 | 0.6483 | 39.74% |
| **Decision Tree** | Raw | 13,908.62 | 15,335.97 | 11,332.25 | 0.6466 | 28.39% |
| **Random Forest** | Raw | 13,034.10 | 14,354.00 | 10,539.41 | 0.6904 | 26.96% |
| **Gradient Boosting (sklearn)** | Raw | 12,686.25 | 14,394.83 | 10,712.47 | 0.6886 | 30.56% |
| **LightGBM** | Log | 13,760.78 | 14,244.06 | 10,424.33 | 0.6951 | 25.09% |
| **XGBoost** | Log | 12,627.91 | 14,269.15 | 10,444.47 | 0.6940 | 26.33% |
| **CatBoost (Winner)** | **Log** | **14,817.27** | **14,162.25** | **10,455.56** | **0.6986** | **25.39%** |

### Key Takeaway:
* **Overfitting Resolved**: Using `RandomizedSearchCV` and early stopping, the massive train-test gaps traditionally seen in gradient boosting were completely eliminated. All models now generalize well with minimal train/test RMSE gaps.
* **Why CatBoost Won**: CatBoost's symmetric trees and native categorical handling restricted overfitting the best, resulting in the lowest Test RMSE of $14,162.25. We serialize this specific model for the underwriting engine.

---

## Project Structure

```text
├── app.py                         # Streamlit Dashboard (Operationalized Model)
├── requirements.txt               # Pinned dependencies
├── run_pipeline.bat               # Batch script to execute the pipeline end-to-end
├── src/
│   ├── eda_and_preprocess.py              # Feature engineering and cleaning
│   ├── Baseline_Model_training.py         # Sklearn baselines (Linear, RF, GB)
│   ├── Advance_Boosting_model_training.py # CatBoost, LightGBM, XGBoost tuning
│   ├── run_full_pipeline.py               # Unified script (Runs all models and saves best)
│   └── Artifact_dir/                      # Automatically generated diagnostic plots
```

---

## How to Run

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone <your-github-repo-url>
cd auto-insurance-claims-modeling

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the unified script to preprocess the data, systematically tune all models, eliminate target leakage, and export the `absolute_best_model.joblib`:
```bash
python src/run_full_pipeline.py
```

### 3. Launch the Dashboard
Once the model is trained, launch the interactive underwriting dashboard:
```bash
streamlit run app.py
```
