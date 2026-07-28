
# ==========================================
# ADVANCED EDA PYTHON IMPLEMENTATION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_by_ai(df):
    """
    Performs Advanced Exploratory Data Analysis on the already loaded dataframe `df`.
    Includes dependency checks, initial inspection, descriptive statistics, 
    correlation analysis, univariate analysis, and multivariate visualizations.
    """
    
    # Set overall aesthetic style for plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.autolayout': True})

    # ==========================================
    # 1. INITIAL DATA INSPECTION
    # ==========================================
    print("--- 1. INITIAL DATA INSPECTION ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print(f"\nDuplicate Rows Count: {df.duplicated().sum()}\n")

    # Identify numerical and categorical columns dynamically
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # ==========================================
    # 2. DESCRIPTIVE STATISTICS
    # ==========================================
    print("--- 2. DESCRIPTIVE STATISTICS ---")
    if num_cols:
        print("\nNumerical Summary (Mean, Std, Min, Percentiles, Max, IQR, Skewness):")
        desc_num = df[num_cols].describe().T
        desc_num['iqr'] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
        desc_num['skewness'] = df[num_cols].skew()
        print(desc_num[['mean', 'std', 'min', '50%', 'max', 'iqr', 'skewness']])
    else:
        print("\nNo numerical columns found for descriptive summary.")

    if cat_cols:
        print("\nCategorical Summary:")
        for col in cat_cols:
            print(f"\nColumn: {col} (Cardinality: {df[col].nunique()})")
            print(df[col].value_counts(normalize=True) * 100)
    else:
        print("\nNo categorical columns found for summary.")

    # ==========================================
    # 3. CORRELATION ANALYSIS
    # ==========================================
    print("\n--- 3. CORRELATION ANALYSIS ---")
    if len(num_cols) > 1:
        corr_matrix_pearson = df[num_cols].corr(method='pearson')
        corr_matrix_spearman = df[num_cols].corr(method='spearman')

        print("\nPearson Correlation Matrix:\n", corr_matrix_pearson)

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
        plt.title("Correlation Matrix Heatmap (Pearson)")
        plt.show()
    else:
        print("\nSkipping correlation analysis: Requires at least 2 numerical columns.")

    # ==========================================
    # 4. UNIVARIATE NUMERICAL & OBJECT ANALYSIS
    # ==========================================
    print("\n--- 4. UNIVARIATE ANALYSIS ---")

    # Numerical Univariate Analysis (Histograms with KDE & Boxplots)
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram + KDE
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Distribution & KDE of {col}')
        
        # Boxplot
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Outlier Detection Boxplot of {col}')
        
        plt.show()

    # Object/Categorical Univariate Analysis (Count Plots)
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
        plt.title(f'Frequency Distribution of Categorical Column: {col}')
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

    # ==========================================
    # 5. MULTIVARIATE ANALYSIS
    # ==========================================
    print("\n--- 5. MULTIVARIATE ANALYSIS ---")

    # Check if specific columns exist before plotting multivariate relationships
    has_sales = 'sales' in df.columns
    has_region = 'region' in df.columns
    has_segment = 'segment' in df.columns

    if has_sales and has_region and has_segment:
        # Multivariate Bar Plot: Sales across Region and Segment (with Hue)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df, 
            x='region', 
            y='sales', 
            hue='segment', 
            estimator=np.mean, 
            errorbar=('ci', 68),  # Updated parameter for confidence intervals in modern seaborn
            palette='muted',
            capsize=0.1
        )
        plt.title('Multivariate Analysis: Average Sales by Region and Segment')
        plt.xlabel('Region')
        plt.ylabel('Average Sales')
        plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Additional Multivariate Analysis: Boxplot grouped by multiple categories
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df, 
            x='region', 
            y='sales', 
            hue='segment', 
            palette='Set3'
        )
        plt.title('Sales Distribution Across Regions grouped by Segments')
        plt.xlabel('Region')
        plt.ylabel('Sales')
        plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    else:
        print("\nSkipping specific multivariate plots: Required columns ('sales', 'region', 'segment') not all present.")
