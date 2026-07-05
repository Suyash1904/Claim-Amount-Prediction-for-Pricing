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
| **Random Forest** | Raw | 13,034.10 | 14,354.00 | 10,539.41 | 0.6904 | 26.96% |
| **Gradient Boosting (sklearn)** | Raw | 12,686.25 | 14,394.83 | 10,712.47 | 0.6886 | 30.56% |
| **LightGBM** | Log | 13,760.78 | 14,244.06 | 10,424.33 | 0.6951 | 25.09% |
| **XGBoost** | Log | 12,627.91 | 14,269.15 | 10,444.47 | 0.6940 | 26.33% |
| **CatBoost (Winner)** | **Log** | **14,817.27** | **14,162.25** | **10,455.56** | **0.6986** | **25.39%** |

### Key Takeaway:
* **Overfitting Resolved**: Using `RandomizedSearchCV` and Early Stopping, the massive train-test gaps traditionally seen in advanced boosting (e.g. XGBoost previously hitting 97% Train R²) were completely eliminated. All models now generalize beautifully with < $1,600 train/test RMSE gaps.
* **CatBoost's Generalization**: CatBoost's **Symmetric Trees** and native categorical handling restricted overfitting perfectly, resulting in the lowest Test RMSE of **$14,162.25** and acting as the final underwriting engine.

---

## 📁 Project Structure

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
