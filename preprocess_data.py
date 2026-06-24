import pandas as pd

# 1. Define File Paths
raw_file_path = 'dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'
new_file_path = 'dataset/Preprocessed_HR_Attrition.csv'

print("Loading raw dataset...")
df = pd.read_csv(raw_file_path)

# 2. Drop Unnecessary (Redundant) Columns
# These columns do not contribute to the prediction process
columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df_clean = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# 3. Convert Target Variable 'Attrition' into Numeric Values
# Machine learning models require numerical data
# Yes = 1 and No = 0
df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
print("'Attrition' column has been converted to 0 and 1.")

# 4. Convert Remaining Categorical (Text) Columns into Numeric Values
# One-Hot Encoding is applied to transform categorical features
# drop_first=True helps avoid the Dummy Variable Trap
df_encoded = pd.get_dummies(df_clean, drop_first=True)
print("Categorical columns have been converted into numeric values using One-Hot Encoding.")

# 5. Save the Preprocessed Data into a New File
df_encoded.to_csv(new_file_path, index=False)

print("\n=======================================================")
print("SUCCESS: Preprocessed data has been saved to a new file!")
print(f"New File Name: {new_file_path}")
print(f"New File Shape (Rows, Columns): {df_encoded.shape}")
print("=======================================================")