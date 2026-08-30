import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Backend set karna zaroori hai taake loop mein graphs overlap na hon
import matplotlib
matplotlib.use('Agg')

print("🧠 Starting Explainable AI (SHAP) Analysis for all datasets...\n")

# Datasets aur unke best models (Random Forest) ki paths
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
    print(f"📊 Generating SHAP explanations for {ds_name}...")
    
    # 1. Data aur Model load karein
    df = pd.read_csv(data_path)
    X = df.drop('Attrition', axis=1)
    model = joblib.load(model_path)
    
    # SHAP calculate karne mein time lagta hai, isliye hum best 1000 random samples lenge
    # (Global features explain karne ke liye 1000 records bohat hote hain)
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    
    # 2. SHAP Tree Explainer initialize karein
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Random Forest 2 classes (Stay/Leave) ka result deta hai, humein Leave (Class 1) chahiye
    if isinstance(shap_values, list):
        shap_values_attrition = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_attrition = shap_values[:, :, 1]
    else:
        shap_values_attrition = shap_values

    # 3. Global Summary Plot (Beeswarm) Banayen
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_attrition, X_sample, show=False, max_display=10)
    
    # Formatting
    plt.title(f'XAI Global Key Drivers of Attrition - {ds_name}', fontsize=14, pad=15)
    plt.tight_layout()
    
    # 4. Graph ko results folder mein save karein
    plot_filename = f'results/SHAP_Global_{ds_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {ds_name} SHAP plot saved to {plot_filename}\n")

print("🎉 ALL XAI PLOTS GENERATED AND SAVED SUCCESSFULLY! 🎉")