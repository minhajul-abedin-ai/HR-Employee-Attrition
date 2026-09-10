import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

print("Starting Explainable AI (SHAP) Analysis for all datasets...\n")

# Results folder ensured
import matplotlib
matplotlib.use('Agg')

print("Starting Explainable AI (SHAP) Analysis for all datasets...\n")

# Datasets and corresponding model paths
datasets_info = {
    'IBM': (
        'dataset/preprocessed/Processed_Advanced_IBM.csv', 
        'models/IBM_Random_Forest.pkl'
    ),
    'KaggleHR': (
        'dataset/preprocessed/Processed_Advanced_KaggleHR.csv', 
        'models/KaggleHR_Random_Forest.pkl'
    ),
    'EmployeeChurn': (
        'dataset/preprocessed/Processed_Advanced_EmployeeChurn.csv', 
        'models/EmployeeChurn_Random_Forest.pkl'
    )
}

for ds_name, (data_path, model_path) in datasets_info.items():
    print(f" Generating SHAP explanations for {ds_name}...")
    
    print(f"Generating SHAP explanations for {ds_name}...")
    
    # 1. Data and Model load

    df = pd.read_csv(data_path)
    X = df.drop('Attrition', axis=1)
    model = joblib.load(model_path)
    
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    
    # 2. SHAP Tree Explainer initialize 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Random Forest 2 classes (Stay/Leave) 
    if isinstance(shap_values, list):
        shap_values_attrition = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_attrition = shap_values[:, :, 1]
    else:
        shap_values_attrition = shap_values

    # 3. Global Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_attrition, X_sample, show=False, max_display=10)
    
    # Formatting
    plt.title(f'XAI Global Key Drivers of Attrition - {ds_name}', fontsize=14, pad=15)
    plt.tight_layout()
    
    plot_filename = f'results/SHAP_Global_{ds_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" {ds_name} SHAP plot saved to {plot_filename}\n")

print(" ALL XAI PLOTS GENERATED AND SAVED SUCCESSFULLY! ")
print(f"{ds_name} SHAP plot saved to {plot_filename}\n")
print("ALL XAI PLOTS GENERATED AND SAVED SUCCESSFULLY! ")
