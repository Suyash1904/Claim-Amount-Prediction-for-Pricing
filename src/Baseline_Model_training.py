import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Custom premium palette
COLOR_PRIMARY = '#10b981' # Emerald Green
COLOR_SECONDARY = '#3b82f6' # Blue
COLOR_ACCENT = '#f59e0b' # Amber
COLOR_DARK = '#1f2937' # Slate Gray

def calculate_mape(y_true, y_pred):
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_evaluate():
    proc_dir = r"c:\Users\SUYASH\Downloads\archive\src\processed"
    artifact_dir = r"C:\Users\SUYASH\Downloads\archive\src\Artifact_dir"
    
    X_train = np.load(os.path.join(proc_dir, "X_train.npy"))
    X_test = np.load(os.path.join(proc_dir, "X_test.npy"))
    y_train = np.load(os.path.join(proc_dir, "y_train.npy"))
    y_test = np.load(os.path.join(proc_dir, "y_test.npy"))
    
    with open(os.path.join(proc_dir, "feature_names.txt"), "r") as f:
        feature_names = f.read().splitlines()
        
    print(f"Loaded train data: {X_train.shape}, test data: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # feature selection via random forest
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    importances = rf_selector.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Feature Selection using Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1).fit(X_train, y_train)
    non_zero_coefs = np.sum(lasso.coef_ != 0)
    
    # Save the feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances[:20], y=sorted_features[:20], hue=sorted_features[:20], legend=False)
    plt.tight_layout()
    plot_path = os.path.join(artifact_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": GridSearchCV(DecisionTreeRegressor(random_state=42), {"max_depth": [3, 5, 7, 10]}, cv=5, n_jobs=-1),
        "Random Forest": GridSearchCV(RandomForestRegressor(random_state=42), {"n_estimators": [100, 200], "max_depth": [5, 10]}, cv=5, n_jobs=-1),
        "Gradient Boosting": GridSearchCV(GradientBoostingRegressor(random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}, cv=5, n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        if isinstance(model, GridSearchCV):
            model.fit(X_train, y_train)
            best_model = model.best_estimator_
            best_score = -model.best_score_
        else:
            model.fit(X_train, y_train)
            best_model = model
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
            best_score = -cv_scores.mean()
            
        trained_models[name] = best_model
        
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        results[name] = {
            "CV_RMSE": best_score,
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
    print("\nBaseline Model Performance:")
    print(results_df.to_string())
    
    results_df.to_csv(os.path.join(proc_dir, "model_comparison.csv"))
    results_df.to_csv(os.path.join(artifact_dir, "model_comparison.csv"))
    
    best_model_name = results_df["Test_RMSE"].idxmin()
    best_model = trained_models[best_model_name]
    
    model_save_path = os.path.join(proc_dir, "best_model.joblib")
    joblib.dump(best_model, model_save_path)
    joblib.dump(best_model, os.path.join(artifact_dir, "best_model.joblib"))
    
    y_test_pred = best_model.predict(X_test)
    residuals = y_test - y_test_pred
    
    plt.figure(figsize=(8, 8))
    # Diagonal line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color=COLOR_ACCENT, linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title(f"{best_model_name} - Actual vs Predicted total_claim_amount", pad=15)
    plt.xlabel("Actual Claim Amount ($)")
    plt.ylabel("Predicted Claim Amount ($)")
    plt.legend()
    plt.tight_layout()
    pred_plot_path = os.path.join(artifact_dir, "actual_vs_predicted.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Actual vs Predicted plot saved at: {pred_plot_path}")
    
    # Plot 2: Residuals Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color=COLOR_PRIMARY, bins=30)
    plt.axvline(0, color=COLOR_ACCENT, linestyle='--', linewidth=1.5)
    plt.title(f"{best_model_name} - Residuals Distribution", pad=15)
    plt.xlabel("Residuals (Actual - Predicted) ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    resid_plot_path = os.path.join(artifact_dir, "residuals_distribution.png")
    plt.savefig(resid_plot_path, dpi=300)
    plt.close()
    print(f"Residuals distribution plot saved at: {resid_plot_path}")
    
    # Print summary performance of winner
    win_metrics = results[best_model_name]
    print(f"\nFinal Winner Metrics ({best_model_name}):")
    print(f"  R-squared: {win_metrics['Test_R2']:.4f}")
    print(f"  RMSE: ${win_metrics['Test_RMSE']:.2f}")
    print(f"  MAE: ${win_metrics['Test_MAE']:.2f}")
    print(f"  MAPE: {win_metrics['Test_MAPE']:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()
