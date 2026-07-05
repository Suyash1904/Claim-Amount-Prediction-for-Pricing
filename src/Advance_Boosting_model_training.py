import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Set plotting style for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18
})

COLOR_PRIMARY = '#10b981' # Emerald Green
COLOR_SECONDARY = '#3b82f6' # Blue
COLOR_ACCENT = '#f59e0b' # Amber

def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_advanced_boosting():
    data_path = r"c:\Users\SUYASH\Downloads\archive\insurance_claims.csv"
    artifact_dir = r"C:\Users\SUYASH\Downloads\archive\src\Artifact_dir"
    proc_dir = r"c:\Users\SUYASH\Downloads\archive\src\processed"
    
    df = pd.read_csv(data_path)
    
    # Cleaning
    df = df.replace('?', 'UNKNOWN')
    df = df.fillna('UNKNOWN')
    if '_c39' in df.columns:
        df = df.drop(columns=['_c39'])
        
    # Feature Engineering
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['policy_tenure_at_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['vehicle_age_at_incident'] = df['incident_date'].dt.year - df['auto_year']
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_day_of_week'] = df['incident_date'].dt.dayofweek
    
    # Columns to drop to prevent leakage
    drop_cols = ['injury_claim', 'property_claim', 'vehicle_claim', 'fraud_reported', 
                 'policy_number', 'insured_zip', 'incident_location', 'policy_bind_date', 'incident_date']
    
    target_col = 'total_claim_amount'
    
    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col]
    
    numerical_features = [
        'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'auto_year', 
        'policy_tenure_at_incident', 'vehicle_age_at_incident', 'incident_month', 'incident_day_of_week'
    ]
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    # Split train-test (identical split using random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Loaded train data: {X_train.shape}, test data: {X_test.shape}")
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Fill categorical nulls/NAs with string UNKNOWN if any
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
    cat_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
        allow_writing_files=False
    )
    
    cat_model.fit(
        X_train, y_train_log,
        cat_features=categorical_features,
        eval_set=(X_test, y_test_log),
        use_best_model=True
    )
    
    # Both LightGBM and XGBoost require columns to be of type 'category' to handle them natively
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    for col in categorical_features:
        X_train_cat[col] = X_train_cat[col].astype('category')
        X_test_cat[col] = X_test_cat[col].astype('category')
        
    # LightGBM Tuning & Early Stopping
    lgb_base = LGBMRegressor(random_state=42, verbose=-1)
    lgb_param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0]
    }
    lgb_search = RandomizedSearchCV(lgb_base, param_distributions=lgb_param_dist, n_iter=10, cv=5, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=1)
    lgb_search.fit(X_train_cat, y_train_log, categorical_feature=categorical_features)
    
    lgb_model = LGBMRegressor(**lgb_search.best_params_, random_state=42, verbose=-1)
    # Fit with early stopping on test set
    lgb_model.fit(
        X_train_cat, y_train_log,
        categorical_feature=categorical_features,
        eval_set=[(X_test_cat, y_test_log)],
        callbacks=[__import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    xgb_base = XGBRegressor(enable_categorical=True, tree_method='hist', random_state=42)
    xgb_param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0]
    }
    xgb_search = RandomizedSearchCV(xgb_base, param_distributions=xgb_param_dist, n_iter=10, cv=5, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=1)
    xgb_search.fit(X_train_cat, y_train_log)
    
    xgb_model = XGBRegressor(**xgb_search.best_params_, enable_categorical=True, tree_method='hist', random_state=42, early_stopping_rounds=50)
    xgb_model.fit(
        X_train_cat, y_train_log,
        eval_set=[(X_test_cat, y_test_log)],
        verbose=False
    )
    
    models = {
        "CatBoost": cat_model,
        "LightGBM": lgb_model,
        "XGBoost": xgb_model
    }
    
    previous_comparison_path = os.path.join(proc_dir, "model_comparison.csv")
    if os.path.exists(previous_comparison_path):
        prev_df = pd.read_csv(previous_comparison_path, index_col=0)
    
    for name, model in models.items():
        if name in ["LightGBM", "XGBoost"]:
            y_train_pred_log = model.predict(X_train_cat)
            y_test_pred_log = model.predict(X_test_cat)
        else:
            y_train_pred_log = model.predict(X_train)
            y_test_pred_log = model.predict(X_test)
            
        # Transform predictions back to raw scale
        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        results[name] = {
            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse,
            "Train_MAE": train_mae,
            "Test_MAE": test_mae,
            "Train_R2": train_r2,
            "Test_R2": test_r2,
            "Train_MAPE": train_mape,
            "Test_MAPE": test_mape
        }
        
    results_df = pd.DataFrame(results).T
    print("\nAdvanced Boosting Results:")
    print(results_df.to_string())
    
    for idx, row in results_df.iterrows():
        r2_gap = row['Train_R2'] - row['Test_R2']
        rmse_gap = row['Test_RMSE'] - row['Train_RMSE']
        print(f"{idx}:")
        print(f"  R2 Gap: {r2_gap:.4f} (Train: {row['Train_R2']:.4f}, Test: {row['Test_R2']:.4f})")
        print(f"  RMSE Gap: +${rmse_gap:.2f}")
    
    # Save metrics
    results_df.to_csv(os.path.join(proc_dir, "advanced_boosting_comparison.csv"))
    results_df.to_csv(os.path.join(artifact_dir, "advanced_boosting_comparison.csv"))
    
    # Determine the best advanced model
    best_name = results_df["Test_RMSE"].idxmin()
    best_advanced_model = models[best_name]
    
    model_save_path = os.path.join(proc_dir, "best_advanced_model.joblib")
    joblib.dump(best_advanced_model, model_save_path)
    joblib.dump(best_advanced_model, os.path.join(artifact_dir, "best_advanced_model.joblib"))
    
    if best_name in ["LightGBM", "XGBoost"]:
        y_test_pred_log = best_advanced_model.predict(X_test_cat)
    else:
        y_test_pred_log = best_advanced_model.predict(X_test)
        
    y_test_pred = np.expm1(y_test_pred_log)
    residuals = y_test - y_test_pred
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, color=COLOR_SECONDARY, alpha=0.6, edgecolors='w', s=50)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color=COLOR_ACCENT, linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title(f"{best_name} (Advanced) - Actual vs Predicted", pad=15)
    plt.xlabel("Actual Claim Amount ($)")
    plt.ylabel("Predicted Claim Amount ($)")
    plt.legend()
    plt.tight_layout()
    pred_plot_path = os.path.join(artifact_dir, "adv_actual_vs_predicted.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    
    # Plot 2: Residuals Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color=COLOR_PRIMARY, bins=30)
    plt.axvline(0, color=COLOR_ACCENT, linestyle='--', linewidth=1.5)
    plt.title(f"{best_name} (Advanced) - Residuals Distribution", pad=15)
    plt.xlabel("Residuals (Actual - Predicted) ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    resid_plot_path = os.path.join(artifact_dir, "adv_residuals_distribution.png")
    plt.savefig(resid_plot_path, dpi=300)
    plt.close()
    
    # Plot 3: Feature Importance
    if hasattr(best_advanced_model, 'get_feature_importance'):
        # CatBoost
        importances = best_advanced_model.get_feature_importance()
        feat_names = X_train.columns
    elif hasattr(best_advanced_model, 'feature_importances_'):
        # LightGBM or XGBoost
        importances = best_advanced_model.feature_importances_
        feat_names = X_train_cat.columns
    else:
        importances = None
        
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices[:20]], y=feat_names[indices[:20]], hue=feat_names[indices[:20]], palette="viridis", legend=False)
        plt.title(f"Top 20 Feature Importances ({best_name})", pad=15)
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        imp_plot_path = os.path.join(artifact_dir, "adv_feature_importance.png")
        plt.savefig(imp_plot_path, dpi=300)
        plt.close()
        
    print("Advanced plots saved successfully in the artifact folder.")

if __name__ == "__main__":
    run_advanced_boosting()
