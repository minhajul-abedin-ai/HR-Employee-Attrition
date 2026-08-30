# Explainable AI (XAI) Framework for Proactive Human Resource Management

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP-purple.svg)

## Project Overview
Employee attrition is a critical operational challenge for modern enterprises, leading to significant financial losses and disrupted workflows. This project presents an end-to-end, highly robust **Explainable Artificial Intelligence (XAI) framework** designed to proactively predict employee turnover. 

Unlike traditional "black-box" machine learning models, this framework integrates **SHAP (SHapley Additive exPlanations)** to provide actionable transparency. It not only predicts *who* is likely to leave but also explains *why*, offering both global (systemic) insights and local (individualized) risk assessments through an interactive **Flask web dashboard**.

## Key Features
*   **Leakage-Free Data Engineering:** Implements strict data partitioning before applying K-Nearest Neighbors (KNN) imputation, Isolation Forest filtering, and SMOTE (Synthetic Minority Over-sampling Technique) to ensure absolute mathematical integrity.
*   **Independent Multi-Domain Evaluation:** Validated independently across three distinct organizational datasets to prove external generalizability.
*   **Explainable AI (XAI):** Uses SHAP `TreeExplainer` to demystify the Random Forest predictions.
*   **Interactive Web Dashboard:** A user-friendly Flask application allowing HR managers to input employee data and instantly receive risk probabilities alongside dynamic visual explanations.

## Datasets Utilized
To ensure robustness, the models were independently trained and evaluated on three heterogeneous datasets (analyzing 26,469 records in total):
1.  **IBM HR Analytics Dataset** (1,470 records, 35 features)
2.  **Kaggle HR Database** (14,999 records, 10 features)
3.  **Employee Churn Records** (10,000 records, 22 features)

## Tech Stack
*   **Programming Language:** Python 3.x
*   **Machine Learning:** Scikit-Learn (Random Forest, Classification Metrics)
*   **Data Imbalance Handling:** Imbalanced-learn (SMOTE)
*   **Interpretability (XAI):** SHAP
*   **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
*   **Web Deployment:** Flask (Python backend), HTML/CSS (Frontend UI)

## Repository Structure
```text
├── dataset/                  # Contains raw and preprocessed .csv files
├── models/                   # Serialized .pkl files (Random Forest, Scalers, Imputers)
├── results/                  # Generated SHAP visualizations and ROC curves
├── templates/                # HTML files for the Flask dashboard UI
├── app.py                    # Main Flask routing script for the web application
├── generate_graphs.py        # Data science script (preprocessing, training, SHAP generation)
├── requirements.txt          # Python dependencies required to run the project
└── README.md                 # Project documentation
Installation & Setup Instructions
Follow these steps to run the project locally on your machine:

1. Clone the repository:

Bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create a Virtual Environment (Recommended):

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies:

Bash
pip install -r requirements.txt
4. Run the Machine Learning Pipeline (Optional):
If you want to retrain the models and generate new SHAP graphs:

Bash
python generate_graphs.py
5. Start the Flask Web Application:

Bash
python app.py
6. Access the Dashboard:
Open your web browser and navigate to: http://127.0.0.1:5000/

Experimental Results
The manually configured Random Forest ensemble demonstrated the most robust performance, avoiding the overfitting issues common with gradient boosting on noisy HR data:

IBM HR Analytics: Accuracy: 0.8673 | ROC-AUC: 0.7458

Kaggle HR Database: Accuracy: 0.9587 | ROC-AUC: 0.9820

Employee Churn Records: Accuracy: 0.7835 | ROC-AUC: 0.4949

Author
Minhajul Abedin

University: Ulster University, London Branch, UK

Course: Computer Science and Technology

Email: Abedin-M1@ulster.ac.uk

License
This project is for academic and research purposes as part of a university dissertation.
