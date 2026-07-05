import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

def run_pipeline():
    data_path = r"c:\Users\SUYASH\Downloads\archive\insurance_claims.csv"
    artifact_dir = r"C:\Users\SUYASH\Downloads\archive\src\Artifact_dir"
    proc_dir = r"c:\Users\SUYASH\Downloads\archive\src\processed"
    os.makedirs(proc_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    df = df.replace('?', 'UNKNOWN')
    df = df.fillna('UNKNOWN')
    if '_c39' in df.columns:
        df = df.drop(columns=['_c39'])
        
    # temporal feature engineering
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['policy_tenure_at_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['vehicle_age_at_incident'] = df['incident_date'].dt.year - df['auto_year']
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_day_of_week'] = df['incident_date'].dt.dayofweek
    
    # check for target leakage before dropping sub-claims
    max_leakage_diff = (df['injury_claim'] + df['property_claim'] + df['vehicle_claim'] - df['total_claim_amount']).abs().max()
    if max_leakage_diff != 0:
        print(f"Warning: sub-claims do not sum to total claim amount. Max diff: {max_leakage_diff}")

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    X_train_std = preprocessor.fit_transform(X_train)
    X_test_std = preprocessor.transform(X_test)
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in categorical_features:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)
        
    X_train_adv = X_train.copy()
    X_test_adv = X_test.copy()
    for col in categorical_features:
        X_train_adv[col] = X_train_adv[col].astype('category')
        X_test_adv[col] = X_test_adv[col].astype('category')
        
    models_dict = {}
    
    lr = LinearRegression()
    lr.fit(X_train_std, y_train)
    models_dict["Linear Regression"] = lr
    
    dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), 
                           {"max_depth": [4, 6, 8, 10], "min_samples_split": [5, 10, 20]}, 
                           cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    dt_grid.fit(X_train_std, y_train)
    models_dict["Decision Tree"] = dt_grid.best_estimator_
    
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), 
                           {"n_estimators": [100, 200], "max_depth": [5, 10]}, 
                           cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train_std, y_train)
    models_dict["Random Forest"] = rf_grid.best_estimator_
    
    gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), 
                           {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 4]}, 
                           cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gb_grid.fit(X_train_std, y_train)
    models_dict["Gradient Boosting"] = gb_grid.best_estimator_
    
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
    lgb_search.fit(X_train_adv, y_train_log)
    
    lgb = LGBMRegressor(**lgb_search.best_params_, random_state=42, verbose=-1)
    lgb.fit(X_train_adv, y_train_log, eval_set=[(X_test_adv, y_test_log)], callbacks=[__import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False)])
    models_dict["LightGBM"] = lgb
    
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
    xgb_search.fit(X_train_adv, y_train_log)
    
    xgb = XGBRegressor(**xgb_search.best_params_, enable_categorical=True, tree_method='hist', random_state=42)
    xgb.fit(X_train_adv, y_train_log, eval_set=[(X_test_adv, y_test_log)], verbose=False)
    models_dict["XGBoost"] = xgb
    
    cat = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, random_seed=42, verbose=False, allow_writing_files=False)
    cat.fit(X_train_cb, y_train_log, cat_features=categorical_features, eval_set=(X_test_cb, y_test_log), use_best_model=True)
    models_dict["CatBoost"] = cat
    
    results = {}
    for name, model in models_dict.items():
        if name in ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
            y_train_pred = model.predict(X_train_std)
            y_test_pred = model.predict(X_test_std)
        elif name in ["LightGBM", "XGBoost"]:
            y_train_pred = np.expm1(model.predict(X_train_adv))
            y_test_pred = np.expm1(model.predict(X_test_adv))
        else:
            y_train_pred = np.expm1(model.predict(X_train_cb))
            y_test_pred = np.expm1(model.predict(X_test_cb))
            
        results[name] = {
            "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "Train_MAE": mean_absolute_error(y_train, y_train_pred),
            "Test_MAE": mean_absolute_error(y_test, y_test_pred),
            "Train_R2": r2_score(y_train, y_train_pred),
            "Test_R2": r2_score(y_test, y_test_pred),
            "Train_MAPE": calculate_mape(y_train, y_train_pred),
            "Test_MAPE": calculate_mape(y_test, y_test_pred)
        }
        
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(proc_dir, "full_model_comparison.csv"))
    results_df.to_csv(os.path.join(artifact_dir, "full_model_comparison.csv"))
    
    best_model_name = results_df["Test_RMSE"].idxmin()
    best_model = models_dict[best_model_name]
    
    joblib.dump(best_model, os.path.join(proc_dir, "absolute_best_model.joblib"))
    joblib.dump(best_model, os.path.join(artifact_dir, "absolute_best_model.joblib"))
    
    if best_model_name in ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
        y_test_pred = best_model.predict(X_test_std)
    elif best_model_name in ["LightGBM", "XGBoost"]:
        y_test_pred = np.expm1(best_model.predict(X_test_adv))
    else:
        y_test_pred = np.expm1(best_model.predict(X_test_cb))
        
    residuals = y_test - y_test_pred
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='w', s=50)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "winner_actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.axvline(0, linestyle='--', linewidth=1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "winner_residuals_distribution.png"), dpi=300)
    plt.close()
    
    if hasattr(best_model, 'get_feature_importance'):
        importances = best_model.get_feature_importance()
        feat_names = X_train.columns
    elif hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        if best_model_name in ["LightGBM", "XGBoost"]:
            feat_names = X_train_adv.columns
        else:
            cat_encoder = preprocessor.named_transformers_['cat']
            encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
            feat_names = numerical_features + encoded_cat_features
    else:
        importances = None
        
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices[:20]], y=feat_names[indices[:20]], hue=feat_names[indices[:20]], legend=False)
        plt.tight_layout()
        imp_plot_path = os.path.join(artifact_dir, "winner_feature_importance.png")
        plt.savefig(imp_plot_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    run_pipeline()
