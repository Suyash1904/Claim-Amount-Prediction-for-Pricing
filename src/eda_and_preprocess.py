import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def run_eda_and_preprocessing():
    data_path = r"c:\Users\SUYASH\Downloads\archive\insurance_claims.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    if '_c39' in df.columns:
        df = df.drop(columns=['_c39'])
        
    # handle missing values represented by '?'
    for col in df.columns:
        num_missing = (df[col] == '?').sum()
        if num_missing > 0:
            print(f"Column '{col}' has {num_missing} missing values represented as '?'")
            
    df = df.replace('?', 'UNKNOWN')
    
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    
    # calculate tenure in days at time of incident
    df['policy_tenure_at_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    
    # temporal features
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_day_of_week'] = df['incident_date'].dt.dayofweek
    
    # verify sub-claims sum perfectly to target. if so, they must be dropped to prevent 100% target leakage
    max_leakage_diff = (df['injury_claim'] + df['property_claim'] + df['vehicle_claim'] - df['total_claim_amount']).abs().max()
    if max_leakage_diff != 0:
        print(f"Warning: Leakage verification failed with diff {max_leakage_diff}")

    leakage_cols = ['injury_claim', 'property_claim', 'vehicle_claim', 'fraud_reported']
    id_cols = ['policy_number', 'insured_zip', 'incident_location']
    date_cols = ['policy_bind_date', 'incident_date']
    
    drop_cols = leakage_cols + id_cols + date_cols
    
    target_col = 'total_claim_amount'
    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col]
    
    numerical_features = [
        'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'auto_year', 
        'policy_tenure_at_incident', 'incident_month', 'incident_day_of_week'
    ]
    
    categorical_features = [col for col in X.columns if col not in numerical_features]
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    print("\n=== Step 4: Train-Test Split ===")
    # We split first to avoid any data leakage during scaling/encoding
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("\n=== Step 5: Encoding and Scaling ===")
    # Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Fit on training set and transform both
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Extract feature names for analysis later
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + encoded_cat_features
    
    print(f"Processed Train Shape: {X_train_proc.shape}")
    print(f"Processed Test Shape: {X_test_proc.shape}")
    print(f"Total features after encoding: {len(feature_names)}")
    
    # Save the processed data for modeling step
    out_dir = r"c:\Users\SUYASH\Downloads\archive\src\processed"
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "X_train.npy"), X_train_proc)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test_proc)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test.values)
    
    # Save feature names and details
    with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
        
    print(f"Preprocessed files saved in {out_dir}")

if __name__ == "__main__":
    run_eda_and_preprocessing()
