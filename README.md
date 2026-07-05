# Auto Insurance Claims Cost Modeling

This repository contains the complete data science pipeline to predict auto insurance claim costs (`total_claim_amount`) at the time of claim notification. It features data preprocessing, feature engineering, target leakage prevention, and a comparative benchmark of standard and advanced machine learning algorithms.

---

## 📋 Table of Contents
* [Project Overview](#-project-overview)
* [Data Quality & Anomalies](#-data-quality--anomalies)
* [Target Leakage Controls](#-target-leakage-controls)
* [Feature Engineering](#-feature-engineering)
* [Model Comparison & Benchmark](#-model-comparison--benchmark)
* [Project Structure](#-project-structure)
* [How to Run](#-how-to-run)

---

## 🔍 Project Overview
Predicting the severity (cost) of an insurance claim at the time it is filed is a core problem in actuarial science and claims management. Accurate cost prediction enables insurance providers to:
1. Optimize pricing structures and adjust policy premiums dynamically.
2. Flag high-cost claims early for senior adjuster routing.
3. Fast-track low-cost, low-risk claims for automatic settlement, reducing administrative overhead.

Our final **CatBoost Regressor** model explains **69.96% of the variance** in claim amounts on unseen test data with a Test RMSE of **$14,139.60** and a Test MAE of **$10,425.88**.

---

## ⚠️ Data Quality & Anomalies
During Exploratory Data Analysis (EDA), we identified a critical data anomaly:
* **Negative Policy Tenure (Policy #794731)**: The claim incident occurred on **2015-02-02**, but the policy was bound on **2015-02-22**, resulting in a tenure of **-20 days**.
* **Implication**: This represents potential policy backdating, a system entry error, or high-risk fraud, which was flagged for risk management.

---

## 🛡️ Target Leakage Controls
A major challenge in claims cost modeling is target leakage. The dataset contains sub-claim breakdowns:
* `injury_claim`
* `property_claim`
* `vehicle_claim`

These columns sum exactly to the target `total_claim_amount`. If included, models achieve a trivial $R^2 = 1.0$, but are completely useless in production because sub-claims are settled *after* the claim has been closed. We **excluded these features** to ensure the model functions as a predictive tool at the time of claim registration. We also excluded `fraud_reported` as fraud status is determined post-incident.

---

## ⚙️ Feature Engineering
To capture risk exposure, we engineered:
* **`policy_tenure_at_incident`**: Days between the policy bind date and the incident date.
* **`vehicle_age_at_incident`**: Years between the incident year and the vehicle's manufacturing year (`auto_year`). Older vehicles are more likely to be declared a total loss at lower caps, while new cars generate higher claim potential.
* **Temporal Attributes**: `incident_month` and `incident_day_of_week` to model seasonality.

---

## 📊 Model Comparison & Benchmark

We evaluated standard algorithms against advanced gradient boosting engines with native categorical support (using a log-transformed target).

| Model | Target Scale | Train RMSE ($) | Test RMSE ($) | Test MAE ($) | Test $R^2$ | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | Raw | 13,316.62 | 15,298.23 | 11,641.45 | 0.6483 | 39.74% |
| **Decision Tree** | Raw | 13,908.62 | 15,335.97 | 11,332.25 | 0.6466 | 28.39% |
| **Random Forest** | Raw | 10,878.05 | 14,356.23 | 10,504.57 | 0.6903 | 27.00% |
| **Gradient Boosting (sklearn)** | Raw | 12,686.25 | 14,394.83 | 10,712.47 | 0.6886 | 30.56% |
| **LightGBM** | Log | 7,040.37 | 15,502.77 | 11,347.30 | 0.6389 | 26.82% |
| **XGBoost** | Log | 4,004.81 | 15,275.09 | 11,373.22 | 0.6494 | 29.31% |
| **CatBoost (Winner)** | **Log** | **14,725.86** | **14,139.60** | **10,425.88** | **0.6996** | **25.33%** |

### Key Takeaway:
* **Overfitting (XGBoost & LightGBM)**: XGBoost memorized the training data (97.72% Train R²), but dropped to 64.94% on the test set due to the small sample size ($1,000$ rows). 
* **CatBoost's Generalization**: CatBoost's **Symmetric Trees** restricted overfitting, while automated **Early Stopping** locked the model at iteration 106, resulting in the lowest Test RMSE of **$14,139.60**.

---

## 📁 Project Structure

```
├── Preprocessing.ipynb            # Jupyter Notebook for EDA & Preprocessing
├── Model_Training.ipynb           # Jupyter Notebook for Model Training & Selection
├── Auto_Insurance_Claims_Presentation.pptx # HR-ready Presentation Slide Deck
├── run_pipeline.bat               # Batch script to execute the pipeline
├── src/
│   ├── eda_and_preprocess.py      # Preprocessing source script
│   ├── Baseline_Model_training.py      # Base training source script
│   ├── Advance_Boosting_model_training.py # CatBoost, LightGBM & XGBoost training script
│   └── run_full_pipeline.py       # Unified source script (Runs all models)
```

---

## 🚀 How to Run

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone <your-github-repo-url>
cd auto-insurance-claims-modeling

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn catboost lightgbm xgboost python-pptx joblib
```

### 2. Execute the Pipeline
Run the unified script to preprocess the data, train all 7 models, and save the absolute winner:
```bash
python src/run_full_pipeline.py
```
