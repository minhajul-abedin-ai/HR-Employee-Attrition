import os
import sqlite3
from flask import Flask, request, render_template, url_for
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Backend par graphs banane ke liye
matplotlib.use('Agg')

app = Flask(__name__)
os.makedirs('static', exist_ok=True)

# Datasets aur Models ki mapping
DATA_CONFIG = {
    'IBM': {
        'csv': 'dataset/preprocessed/Processed_Advanced_IBM.csv',
        'model': 'models/IBM_Random_Forest.pkl'
    },
    'KaggleHR': {
        'csv': 'dataset/preprocessed/Processed_Advanced_KaggleHR.csv',
        'model': 'models/KaggleHR_Random_Forest.pkl'
    },
    'EmployeeChurn': {
        'csv': 'dataset/preprocessed/Processed_Advanced_EmployeeChurn.csv',
        'model': 'models/EmployeeChurn_Random_Forest.pkl'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

# YEH ROUTE MISSING THA JO AB ADD KAR DIYA HAI
@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        emp_id_str = request.form['emp_id']
        dataset_choice = request.form['dataset_choice']
        
        try:
            emp_index = int(emp_id_str)
            
            # Load selected dataset and model
            config = DATA_CONFIG[dataset_choice]
            df = pd.read_csv(config['csv'])
            model = joblib.load(config['model'])
            
            # Check if emp_index is valid
            if emp_index < 0 or emp_index >= len(df):
                return render_template('index.html', result=f"Error: Employee ID must be between 0 and {len(df)-1} for {dataset_choice}.", prob=0, emp_id=emp_id_str)

            # Separate Features
            X = df.drop('Attrition', axis=1)
            
            # Get specific employee data
            employee_features = X.iloc[[emp_index]]
            
            # Prediction
            prediction = model.predict(employee_features)[0]
            probability = model.predict_proba(employee_features)[0][1] * 100
            
            result_text = "High Risk (Likely to Leave)" if prediction == 1 else "Low Risk (Likely to Stay)"

            # SHAP (Explainable AI) Graph Generation for specific employee
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(employee_features)
            
            if isinstance(shap_values, list):
                shap_val_single = shap_values[1][0]
            else:
                shap_val_single = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

            # Fix base value for waterfall plot
            raw_expected_value = explainer.expected_value
            if hasattr(raw_expected_value, '__len__') and type(raw_expected_value) is not str:
                base_value = float(raw_expected_value[1]) 
            else:
                base_value = float(raw_expected_value)

            shap_val_single = np.array(shap_val_single, dtype=float).flatten()

            plt.figure(figsize=(8, 4.5))
            shap.plots._waterfall.waterfall_legacy(
                base_value, shap_val_single, feature_names=X.columns, max_display=8, show=False
            )
            plt.title(f'Decision Factors for Employee #{emp_index} ({dataset_choice})', fontsize=12, pad=15)
            plt.tight_layout()
            
            plot_filename = f'shap_emp_{dataset_choice}_{emp_index}.png'
            plt.savefig(os.path.join('static', plot_filename), dpi=150, bbox_inches='tight')
            plt.close()

            return render_template('index.html', result=result_text, prob=round(probability, 2), emp_id=emp_index, plot_url=url_for('static', filename=plot_filename))
            
        except Exception as e:
            return render_template('index.html', result=f"Error: {str(e)}", prob=0, emp_id=emp_id_str)

if __name__ == '__main__':
    app.run(debug=True)