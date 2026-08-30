# Explainable AI Framework for Employee Attrition Prediction

An Explainable Artificial Intelligence (XAI) project for **predicting employee attrition and identifying the factors that influence employee turnover** across multiple HR datasets.

The project compares seven machine-learning classifiers, applies a leakage-aware preprocessing pipeline, evaluates model performance, uses **SHAP** for global and local explanations, and provides a **Flask web dashboard** for employee-level attrition-risk analysis.

## Project Objectives

The main objectives are to:

- Predict whether an employee is likely to leave an organisation.
- Compare multiple machine-learning algorithms across different HR datasets.
- Handle missing values, outliers, feature scaling, and class imbalance appropriately.
- Use Explainable AI to identify the most influential attrition drivers.
- Provide employee-level explanations through SHAP waterfall plots.
- Demonstrate the prediction system through a simple Flask web application.

## Datasets

The project uses three employee attrition/churn datasets:

| Dataset | Raw Records | Raw Features/Columns | Processed Columns |
|---|---:|---:|---:|
| IBM HR Analytics | 1,470 | 35 | 45 |
| Kaggle HR | 14,999 | 9 | 17 |
| Employee Churn | 10,000 | 22 | 32 |

During preprocessing, the different target variables are converted to a common binary target named **`Attrition`**.

## Machine-Learning Models

Seven classification algorithms are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. XGBoost
7. Support Vector Machine (SVM)

The deployment pipeline uses a **Random Forest** model for each dataset.

## Data-Processing Pipeline

The advanced training pipeline in `model_training.py` follows these stages:

1. Load the processed dataset.
2. Split data into **80% training and 20% testing** using `random_state=42`.
3. Apply **KNN Imputation** using only the training data.
4. Detect and remove training-set outliers using **Isolation Forest**.
5. Apply **StandardScaler**, fitted only on the training data.
6. Apply **SMOTE** only to the cleaned training data to address class imbalance.
7. Train and compare seven machine-learning models.
8. Evaluate predictions using Accuracy and ROC-AUC.
9. Save the Random Forest model, scaler, and imputer for each dataset.

Keeping imputation, scaling, outlier handling, and SMOTE after the train/test split helps reduce data leakage.

## Explainable AI

The project uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

### Global explanations

Global SHAP summary/beeswarm plots show which features have the largest overall influence on attrition predictions for each dataset.

Generated outputs include:

- `results/SHAP_Global_IBM.png`
- `results/SHAP_Global_KaggleHR.png`
- `results/SHAP_Global_EmployeeChurn.png`

### Local explanations

The Flask dashboard generates an individual **SHAP waterfall plot** for a selected employee record. This helps show which employee characteristics push the model towards a higher or lower predicted attrition risk.

## Recorded Model Results

The existing project results include the following Random Forest performance:

| Dataset | Accuracy | ROC-AUC |
|---|---:|---:|
| IBM | 0.8915 | 0.7950 |
| Kaggle HR | 0.9587 | 0.9820 |
| Employee Churn | 0.8125 | 0.5210 |

The complete recorded comparison is available in:

```text
results/Model_Comparison_Results.csv
```

> Note: `generate_graphs.py` contains the recorded experimental Accuracy/AUC values used to reproduce the report figures. Re-running a different training configuration may produce different values.

## Web Application

The project includes a Flask-based interface named **ALMAS**.

The dashboard allows a user to:

- Select one of the three HR datasets.
- Enter an employee row/index.
- Generate an attrition-risk prediction.
- View the predicted probability.
- View a local SHAP explanation for the selected employee.
- Open an administration page describing the available data pipelines.

Main routes:

```text
/          Main XAI dashboard
/admin     Administration page
/predict   Prediction endpoint
```

## Project Structure

```text
Minhajul/
│
├── app.py                         # Flask XAI prediction dashboard
├── check.py                       # Displays raw dataset dimensions/columns
├── preprocess_data.py             # Encoding and common target preparation
├── model_training.py              # Advanced leakage-aware ML pipeline
├── train_evaluate.py              # Direct seven-model comparison/baseline script
├── generate_xai.py                # Global SHAP generation script
├── generate_graphs.py             # Research-result charts and aligned SHAP/ROC figures
├── hr_system.db                   # Local database file
│
├── dataset/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── Dataset2_HR_comma_sep.csv
│   ├── employee_churn_dataset.csv
│   └── preprocessed/
│       ├── Processed_Advanced_IBM.csv
│       ├── Processed_Advanced_KaggleHR.csv
│       └── Processed_Advanced_EmployeeChurn.csv
│
├── models/
│   ├── IBM_*.pkl
│   ├── KaggleHR_*.pkl
│   ├── EmployeeChurn_*.pkl
│   ├── *_imputer.pkl
│   └── *_scaler.pkl
│
├── results/
│   ├── Model_Comparison_Results.csv
│   ├── Accuracy_Comparison_BarChart.png
│   ├── Best_Models_ROC_Curve.png
│   ├── SHAP_Global_IBM.png
│   ├── SHAP_Global_KaggleHR.png
│   ├── SHAP_Global_EmployeeChurn.png
│   ├── Architecture.png
│   └── flask_dashboard.png
│
├── static/                         # Dynamically generated local SHAP plots
└── templates/
    ├── index.html
    └── admin.html
```

## Requirements

Recommended: **Python 3.10+**

Install the main dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap flask joblib
```

Main libraries used:

- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Flask
- Joblib

## How to Run

Open a terminal inside the project directory.

### 1. Check the datasets

```bash
python check.py
```

### 2. Preprocess all datasets

```bash
python preprocess_data.py
```

This creates the three processed datasets inside:

```text
dataset/preprocessed/
```

### 3. Run the advanced training pipeline

```bash
python model_training.py
```

This performs train/test splitting, KNN imputation, Isolation Forest filtering, scaling, SMOTE, model training, and saves the deployment Random Forest models and preprocessing objects.

### 4. Generate research graphs and SHAP figures

```bash
python generate_graphs.py
```

The generated figures are saved inside:

```text
results/
```

### 5. Run the Flask application

```bash
python app.py
```

Then open the local Flask address shown in the terminal, normally:

```text
http://127.0.0.1:5000/
```

## Alternative Baseline Evaluation

`train_evaluate.py` is a separate direct model-comparison script. It trains all seven models on the encoded processed datasets and exports Accuracy, Precision, Recall, F1-score, ROC curves, and model files.

Run it with:

```bash
python train_evaluate.py
```

Because this script saves model files using some of the same filenames as the advanced pipeline, it should be treated as an **alternative experiment**, not as an additional step after `model_training.py`.

## Important Implementation Note

The advanced models created by `model_training.py` are trained after KNN imputation and feature scaling, and corresponding `*_imputer.pkl` and `*_scaler.pkl` objects are saved.

For completely consistent deployment on new/unscaled records, inference should apply the saved imputer and scaler before passing features to those trained models. The current `app.py` directly passes rows from the processed CSV to the Random Forest model, so this is a current implementation limitation that should be addressed before production deployment.

## Reproducibility

Where supported, the project uses:

```python
random_state = 42
```

This improves reproducibility for train/test splitting, model training, SMOTE, Isolation Forest, and sampling.

Model `.pkl` files are Python/scikit-learn version dependent. If loading an existing model produces a version warning, retraining the models in the current environment is recommended.

## Ethical Use

Employee attrition predictions should be used as **decision-support information**, not as the sole basis for employment decisions. Predictions may reflect limitations or biases in the underlying datasets. Human review, fairness testing, privacy protection, and organisational context are important when applying this type of system in practice.

## Author

**Minhajul Abedin**

MSc Computer Science and Technology  
Ulster University, London Campus

## Academic Project Title

**An Explainable AI (XAI) Framework for Proactive Human Resource Management: Predicting Employee Attrition and Uncovering Key Drivers**
