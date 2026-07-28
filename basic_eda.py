
import pandas as pd
import numpy as np

def perform_eda(df: pd.DataFrame):
    """
    Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    
    Returns:
    None (Prints the analysis directly to the console).
    """
    print("="*60)
    print("📊 BASIC EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # 1. Dataset Dimensions
    print("\n[1] DATASET DIMENSIONS")
    print(f"Number of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")
    
    # 2. Columns & Data Types
    print("\n[2] COLUMNS AND DATA TYPES")
    dtype_df = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.notnull().sum().values
    }).reset_index(drop=True)
    print(dtype_df.to_string(index=False))
    
    # 3. Missing Values Summary
    print("\n[3] MISSING VALUES SUMMARY")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_df.empty:
        print("🎉 Great news! There are no missing values in this dataset.")
    else:
        print(missing_df.to_string())
        
    # 4. Duplicate Rows
    print("\n[4] DUPLICATE ROWS")
    duplicate_count = df.duplicated().sum()
    duplicate_percent = (duplicate_count / len(df)) * 100
    print(f"Number of duplicate rows: {duplicate_count} ({duplicate_percent:.2f}% of total data)")
    
    # 5. Statistical Summary for Numerical Columns
    print("\n[5] NUMERICAL FEATURES SUMMARY")
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        print(num_cols.describe().T.to_string())
    else:
        print("No numerical columns found in the dataset.")
        
    # 6. Statistical Summary for Categorical Columns
    print("\n[6] CATEGORICAL FEATURES SUMMARY")
    cat_cols = df.select_dtypes(include=['object', 'category'])
    if not cat_cols.empty:
        print(cat_cols.describe().T.to_string())
    else:
        print("No categorical columns found in the dataset.")
        
    print("\n" + "="*60)
    print("END OF EDA")
    print("="*60)

# ==========================================
# EXAMPLE USAGE:
# ==========================================
if __name__ == "__main__":
    # Creating a sample DataFrame for demonstration
    data = {
        'age': [25, 32, np.nan, 47, 51, 32],
        'salary': [50000, 60000, 120000, np.nan, 85000, 60000],
        'department': ['HR', 'IT', 'Finance', 'IT', np.nan, 'HR'],
        'is_remote': [True, False, True, False, True, False]
    }
    
    sample_df = pd.DataFrame(data)
    
    # Run the function
    perform_eda(sample_df)
