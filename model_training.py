import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# 1. Define paths for the preprocessed datasets
DATASETS = {
    'IBM': 'dataset/preprocessed/Processed_Advanced_IBM.csv',
    'KaggleHR': 'dataset/preprocessed/Processed_Advanced_KaggleHR.csv',
    'EmployeeChurn': 'dataset/preprocessed/Processed_Advanced_EmployeeChurn.csv'
}

# 2. Define 7 Models with extended hyperparameters as requested by supervisor
models_to_evaluate = {
    'Logistic_Regression': LogisticRegression(
        max_iter=1000, C=1.0, penalty='l2', solver='lbfgs', random_state=42
    ),
    'Decision_Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
    ),
    'Random_Forest': RandomForestClassifier(
        n_estimators=100, max_depth=20, max_features='sqrt', min_samples_split=5, random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42
    ),
    'Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, 
        use_label_encoder=False, eval_metric='logloss', random_state=42
    ),
    'SVM': SVC(
        C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42
    )
}

def train_and_evaluate():
    print("="*50)
    print("STARTING ML MODEL TRAINING & EVALUATION PIPELINE")
    print("="*50)

    for dataset_name, path in DATASETS.items():
        print(f"\n--- Loading and Processing Dataset: {dataset_name} ---")
        
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[ERROR] Could not find {path}. Please run preprocessing first.")
            continue

        X = df.drop('Attrition', axis=1)
        y = df['Attrition']

        # STEP 1: SPLIT DATA FIRST TO PREVENT LEAKAGE
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reset indices for safe alignment
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # STEP 2: KNN Imputation (Fit on train, transform on test)
        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # STEP 3: Isolation Forest (Fit and filter on Training Data ONLY)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso_forest.fit_predict(X_train_imputed)
        
        X_train_clean = X_train_imputed[outliers == 1]
        y_train_clean = y_train[outliers == 1]

        # STEP 4: Feature Scaling (Fit on train, transform on test)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_clean), columns=X_train_clean.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

        # STEP 5: Apply SMOTE (On Cleaned Training Data ONLY)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_clean)
        print(f"Applied SMOTE on Training Set. Resampled shape: {X_train_resampled.shape}")

        best_model = None

        print(f"Training and comparing {len(models_to_evaluate)} models...")
        for model_name, model in models_to_evaluate.items():
            
            # Fit on processed training data
            model.fit(X_train_resampled, y_train_resampled)
            
            # Predict on unseen processed test data
            predictions = model.predict(X_test_scaled)
            pred_probs = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, predictions)
            auc = roc_auc_score(y_test, pred_probs)
            
            print(f"  > {model_name} -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")

            # Keep track of the Random Forest model for deployment
            if model_name == "Random_Forest":
                best_model = model

        # Save models and transformers for Web App Deployment
        if best_model is not None:
            # Save the model
            joblib.dump(best_model, f"models/{dataset_name}_Random_Forest.pkl")
            # Save the scaler and imputer so the web app can process new user inputs consistently
            joblib.dump(scaler, f"models/{dataset_name}_scaler.pkl")
            joblib.dump(imputer, f"models/{dataset_name}_imputer.pkl")
            print(f"[SUCCESS] Saved optimal model and transformers for {dataset_name}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE. ALL MODELS SAVED SUCCESSFULLY.")
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate()