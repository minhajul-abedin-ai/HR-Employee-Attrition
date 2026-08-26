import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# Create output folder
OUTPUT_FOLDER = 'dataset/preprocessed'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_dataset(file_path, target_col, dataset_name, columns_to_drop=None):
    print(f"\n{'='*50}")
    print(f"Starting Preprocessing for: {dataset_name}")
    print(f"{'='*50}")

    try:
        # Load Data
        df = pd.read_csv(file_path)
        print(f"Original Data Shape: {df.shape}")

        # Drop unnecessary ID columns
        if columns_to_drop:
            existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=existing_cols_to_drop, axis=1)

        # Separate target from features to prevent One-Hot Encoding the target
        y_raw = df[target_col]
        X_raw = df.drop(target_col, axis=1)

        # Label Encode ONLY the target column (e.g., 'Yes'/'No' to 1/0)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name='Attrition')

        # Apply One-Hot Encoding to all other categorical features
        categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns
        print(f"Applying One-Hot Encoding to: {list(categorical_cols)}")
        X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

        print("NOTE: KNN Imputation, Isolation Forest, and Scaling are intentionally skipped here.")
        print("They will be applied strictly AFTER train/test splitting in model_training.py to prevent Data Leakage.")

        # Save Processed Dataset (Features and Target combined)
        final_df = pd.concat([X_encoded.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        
        # Kept the same naming convention so it doesn't break other files
        output_name = os.path.join(OUTPUT_FOLDER, f'Processed_Advanced_{dataset_name}.csv')
        final_df.to_csv(output_name, index=False)
        print(f"[SUCCESS] Saved as {output_name}")

    except Exception as e:
        print(f"[ERROR] Error processing {dataset_name}: {e}")

# ==========================================
# EXECUTE PIPELINE FOR ALL 3 DATASETS
# ==========================================

# 1. IBM Dataset 
preprocess_dataset(
    file_path='dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv',
    target_col='Attrition',
    dataset_name='IBM',
    columns_to_drop=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
)

# 2. Kaggle HR Dataset 
preprocess_dataset(
    file_path='dataset/Dataset2_HR_comma_sep.csv',
    target_col='left',
    dataset_name='KaggleHR',
    columns_to_drop=['employee_id']
)

# 3. Employee Churn Dataset 
preprocess_dataset(
    file_path='dataset/employee_churn_dataset.csv',
    target_col='Churn',
    dataset_name='EmployeeChurn',
    columns_to_drop=['Employee ID']
)

print("\n[COMPLETED] ALL 3 DATASETS PREPROCESSED SUCCESSFULLY.")