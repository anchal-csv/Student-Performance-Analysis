import os
import sqlite3
import pandas as pd
import numpy as np

def load_data_from_db(db_path="dataset/student_database.db"):
    """Loads student dataset from SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    # Read the table, dropping the primary key column 'id' to get the original schema
    df = pd.read_sql_query("SELECT Student_ID, Name, Gender, Age, Department, Attendance_Percentage, Study_Hours, Sleep_Hours, Internet_Usage_Hours, Previous_Grades, Family_Income, Parent_Education, Final_Exam_Marks, Result FROM students", conn)
    conn.close()
    return df

def load_data_from_csv(csv_path="dataset/student_performance.csv"):
    """Loads student dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    return pd.read_csv(csv_path)

def clean_data(df):
    """
    Cleans and preprocesses the student dataset:
    - Removes duplicates based on Student_ID.
    - Handles missing values in Sleep_Hours and Family_Income.
    - Enforces proper data types.
    """
    print("--- Starting Data Cleaning Process ---")
    print(f"Original shape: {df.shape}")
    
    # 1. Handle Duplicates
    num_duplicates = df.duplicated(subset=["Student_ID"]).sum()
    print(f"Found {num_duplicates} duplicate records based on Student_ID.")
    df_cleaned = df.drop_duplicates(subset=["Student_ID"], keep="first").copy()
    print(f"Shape after removing duplicates: {df_cleaned.shape}")
    
    # 2. Handle Missing Values
    # Check for missing values before cleaning
    null_summary = df_cleaned.isnull().sum()
    print("\nMissing values count before imputation:")
    for col, count in null_summary.items():
        if count > 0:
            print(f"  - {col}: {count}")
            
    # Impute Sleep_Hours with median
    if "Sleep_Hours" in df_cleaned.columns and df_cleaned["Sleep_Hours"].isnull().any():
        sleep_median = df_cleaned["Sleep_Hours"].median()
        df_cleaned["Sleep_Hours"] = df_cleaned["Sleep_Hours"].fillna(sleep_median)
        print(f"Imputed missing Sleep_Hours with median: {sleep_median:.1f}")
        
    # Impute Family_Income with mode
    if "Family_Income" in df_cleaned.columns and df_cleaned["Family_Income"].isnull().any():
        income_mode = df_cleaned["Family_Income"].mode()[0]
        df_cleaned["Family_Income"] = df_cleaned["Family_Income"].fillna(income_mode)
        print(f"Imputed missing Family_Income with mode: '{income_mode}'")
        
    # Double check missing values
    remaining_nulls = df_cleaned.isnull().sum().sum()
    print(f"Remaining missing values: {remaining_nulls}")
    
    # 3. Enforce Proper Data Types
    dtype_dict = {
        "Student_ID": str,
        "Name": str,
        "Gender": str,
        "Age": int,
        "Department": str,
        "Attendance_Percentage": float,
        "Study_Hours": float,
        "Sleep_Hours": float,
        "Internet_Usage_Hours": float,
        "Previous_Grades": float,
        "Family_Income": str,
        "Parent_Education": str,
        "Final_Exam_Marks": float,
        "Result": str
    }
    
    # Apply type casting
    for col, dtype in dtype_dict.items():
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(dtype)
            
    print("Data types validated successfully.")
    print("--- Data Cleaning Process Completed ---\n")
    return df_cleaned

if __name__ == "__main__":
    # If run as script, load, clean and save
    try:
        # Load from SQLite database to demonstrate DB integration
        df_raw = load_data_from_db()
        df_cleaned = clean_data(df_raw)
        
        # Save cleaned dataset
        cleaned_csv_path = "dataset/student_performance_cleaned.csv"
        df_cleaned.to_csv(cleaned_csv_path, index=False)
        print(f"Cleaned dataset saved successfully to {cleaned_csv_path}")
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
