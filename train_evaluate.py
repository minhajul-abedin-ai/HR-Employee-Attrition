import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib  # Model save karne ke liye

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Results aur Models ke folders ensure karein
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)  # NAYA: Models folder

# Datasets ki list
datasets_info = {
    'IBM': 'dataset/preprocessed/Processed_Advanced_IBM.csv',
    'KaggleHR': 'dataset/preprocessed/Processed_Advanced_KaggleHR.csv',
    'EmployeeChurn': 'dataset/preprocessed/Processed_Advanced_EmployeeChurn.csv'
}

# 7 Models define karein
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results_list = []
roc_data = {}

print("🚀 Starting Model Training, Evaluation & Saving...\n")

for ds_name, path in datasets_info.items():
    print(f"📊 Processing Dataset: {ds_name}...")
    df = pd.read_csv(path)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model_name = None
    best_roc_auc = 0
    best_fpr = None
    best_tpr = None

    for model_name, model in models.items():
        # 1. Model Train Karein
        model.fit(X_train, y_train)
        
        # 2. MODEL SAVE KAREIN (.pkl file mein)
        model_filename = f"models/{ds_name}_{model_name}.pkl"
        joblib.dump(model, model_filename)
        
        # 3. Predict & Metrics Calculation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC AUC preparation (for best model in each dataset)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = model_name
                best_fpr = fpr
                best_tpr = tpr
                
        # Store results
        results_list.append({
            'Dataset': ds_name,
            'Model': model_name.replace('_', ' '),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
    roc_data[ds_name] = {'FPR': best_fpr, 'TPR': best_tpr, 'AUC': best_roc_auc, 'Model': best_model_name.replace('_', ' ')}
    print(f"✅ {ds_name} completed! All 7 models saved. Best Model: {best_model_name.replace('_', ' ')} (AUC: {best_roc_auc:.4f})\n")

# ==========================================
# SAVE RESULTS TO CSV (For Research Paper Table)
# ==========================================
results_df = pd.DataFrame(results_list)
results_df.to_csv('results/Model_Comparison_Results.csv', index=False)
print("💾 Results saved to 'results/Model_Comparison_Results.csv'")

# ==========================================
# PLOT: ACCURACY COMPARISON BAR CHART
# ==========================================
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")
sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Dataset', palette='viridis')
plt.title('Model Accuracy Comparison Across 3 Datasets', fontsize=16, pad=15)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xlabel('Machine Learning Models', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/Accuracy_Comparison_BarChart.png', dpi=300)
print("📈 Accuracy Bar Chart saved to 'results/Accuracy_Comparison_BarChart.png'")

# ==========================================
# PLOT: ROC CURVES FOR BEST MODELS
# ==========================================
plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red']

for idx, (ds_name, data) in enumerate(roc_data.items()):
    plt.plot(data['FPR'], data['TPR'], color=colors[idx], lw=2, 
             label=f"{ds_name} - {data['Model']} (AUC = {data['AUC']:.3f})")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) - Best Models', fontsize=16, pad=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('results/Best_Models_ROC_Curve.png', dpi=300)
print("📉 ROC Curves saved to 'results/Best_Models_ROC_Curve.png'")
print("\n🎉 ALL TRAINING, SAVING AND PLOTTING COMPLETED SUCCESSFULLY! 🎉")