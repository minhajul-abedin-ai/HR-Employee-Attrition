import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# Results folder banayen (agar nahi hai)
os.makedirs('results', exist_ok=True)

print("="*50)
print("GENERATING NEW BEAUTIFUL & CLEAN GRAPHS...")
print("="*50)

# 1. HARDCODED TERMINAL RESULTS FOR CSV AND BAR CHART 
data = [
    # IBM
    ['IBM', 'Logistic_Regression', 0.7891, 0.7781],
    ['IBM', 'Decision_Tree', 0.8231, 0.5498],
    ['IBM', 'Random_Forest', 0.8915, 0.7950], 
    ['IBM', 'AdaBoost', 0.8707, 0.7777],
    ['IBM', 'Gradient_Boosting', 0.8741, 0.7887],
    ['IBM', 'XGBoost', 0.8878, 0.7890],
    ['IBM', 'SVM', 0.8673, 0.7779],
    # KaggleHR
    ['KaggleHR', 'Logistic_Regression', 0.6713, 0.7342],
    ['KaggleHR', 'Decision_Tree', 0.9493, 0.9685],
    ['KaggleHR', 'Random_Forest', 0.9587, 0.9820], 
    ['KaggleHR', 'AdaBoost', 0.8783, 0.9413],
    ['KaggleHR', 'Gradient_Boosting', 0.9440, 0.9771],
    ['KaggleHR', 'XGBoost', 0.9490, 0.9800],
    ['KaggleHR', 'SVM', 0.9047, 0.9433],
    # EmployeeChurn
    ['EmployeeChurn', 'Logistic_Regression', 0.5180, 0.5171],
    ['EmployeeChurn', 'Decision_Tree', 0.7520, 0.4868],
    ['EmployeeChurn', 'Random_Forest', 0.8125, 0.5210], 
    ['EmployeeChurn', 'AdaBoost', 0.6715, 0.4913],
    ['EmployeeChurn', 'Gradient_Boosting', 0.8020, 0.5094],
    ['EmployeeChurn', 'XGBoost', 0.8000, 0.5048],
    ['EmployeeChurn', 'SVM', 0.6955, 0.5039],
]

df_results = pd.DataFrame(data, columns=['Dataset', 'Model', 'Accuracy', 'AUC'])
df_results.to_csv('results/Model_Comparison_Results.csv', index=False)
print("[SUCCESS] Model_Comparison_Results.csv generated.")

# Plot Bar Chart
plt.figure(figsize=(14, 8)) 
sns.barplot(data=df_results, x='Model', y='Accuracy', hue='Dataset', palette='viridis')
plt.title('Model Accuracy Comparison Across Datasets', fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=12) 
plt.yticks(fontsize=12)
plt.ylim(0, 1.1)
plt.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig('results/Accuracy_Comparison_BarChart.png', dpi=300) 
plt.close()
print("[SUCCESS] Accuracy_Comparison_BarChart.png generated.")

# 2. ROC CURVES & HIGH-QUALITY SHAP PLOTS
datasets_info = {
    'IBM': 'dataset/preprocessed/Processed_Advanced_IBM.csv',
    'KaggleHR': 'dataset/preprocessed/Processed_Advanced_KaggleHR.csv',
    'EmployeeChurn': 'dataset/preprocessed/Processed_Advanced_EmployeeChurn.csv'
}

fig_roc, ax_roc = plt.subplots(figsize=(12, 9))

for ds_name, path in datasets_info.items():
    print(f"Processing ROC and SHAP for {ds_name}...")
    try:
        df = pd.read_csv(path)
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        scaler = joblib.load(f"models/{ds_name}_scaler.pkl")
        imputer = joblib.load(f"models/{ds_name}_imputer.pkl")
        rf_model = joblib.load(f"models/{ds_name}_Random_Forest.pkl")
        
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)
        
        y_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2.5, label=f'{ds_name} (AUC = {roc_auc:.4f})')
        
        # --- NEW CLEAN SHAP BEESWARM PLOT GENERATION ---
        explainer = shap.TreeExplainer(rf_model)
        X_shap_sample = X_test_scaled.sample(n=min(300, len(X_test_scaled)), random_state=42)
        shap_values = explainer.shap_values(X_shap_sample)
        
        if isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[1]
        else:
            shap_vals_to_plot = shap_values
            
        # Failsafe: if SHAP returned 3D interaction matrix, convert it to 2D
        if len(np.shape(shap_vals_to_plot)) == 3:
            shap_vals_to_plot = np.sum(shap_vals_to_plot, axis=2)
            
        plt.figure(figsize=(10, 6)) # Optimal minimalist size
        
        # plot_type="dot" forces the standard, beautiful beeswarm plot
        shap.summary_plot(
            shap_vals_to_plot, 
            X_shap_sample, 
            plot_type="dot", 
            max_display=10,  # Limits to top 10 features to keep it clean and minimal
            show=False
        )
        
        plt.title(f"Key Predictive Factors - {ds_name}", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(f"results/SHAP_Global_{ds_name}.png", dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close()
        print(f"[SUCCESS] SHAP_Global_{ds_name}.png generated.")
        
    except Exception as e:
        print(f"[ERROR] Failed processing {ds_name}: {e}")

# Finalize ROC plot formatting
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontsize=14)
ax_roc.set_ylabel('True Positive Rate', fontsize=14)
ax_roc.set_title('Receiver Operating Characteristic (ROC) - Optimal Models', fontsize=16, pad=15)
ax_roc.legend(loc="lower right", fontsize=12)
ax_roc.tick_params(axis='both', which='major', labelsize=12)
fig_roc.tight_layout()
fig_roc.savefig('results/Best_Models_ROC_Curve.png', dpi=300, facecolor='white')
print("[SUCCESS] Best_Models_ROC_Curve.png generated.")

print("\n" + "="*50)
print("ALL NEW ALIGNED & MINIMALIST GRAPHS GENERATED SUCCESSFULLY!")
print("="*50)