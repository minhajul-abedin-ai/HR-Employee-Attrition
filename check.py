import pandas as pd

datasets = {
    "Dataset 1 (IBM)": "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "Dataset 2 (Kaggle HR)": "dataset/Dataset2_HR_comma_sep.csv",
    "Dataset 3 (Employee Churn)": "dataset/employee_churn_dataset.csv"
}

for name, path in datasets.items():
    print(f"\n{'='*40}")
    print(f"🔍 Checking {name}")
    print(f"{'='*40}")
    try:
        df = pd.read_csv(path)
        print(f"Total Rows: {df.shape[0]}")
        print(f"Total Columns: {df.shape[1]}\n")
        print("Column Names:")
        print(df.columns.tolist())
    except Exception as e:
        print(f"❌ Error loading file: {e}")