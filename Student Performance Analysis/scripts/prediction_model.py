import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_evaluate_models(csv_path="dataset/student_performance_cleaned.csv", model_dir="dashboard"):
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Define features and target
    features = ['Attendance_Percentage', 'Study_Hours', 'Sleep_Hours', 'Previous_Grades']
    target = 'Final_Exam_Marks'
    
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    }
    
    best_r2 = -1.0
    best_model_name = ""
    best_model = None
    results = {}
    
    print("--- Training and Evaluating Models ---")
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
        
        print(f"\n{name} Results:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Squared Error (MSE):  {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  R-squared Score (R2):      {r2:.4f}")
        
        # Check if it's the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with R2 Score of {best_r2:.4f}")
    
    # Save the best model
    model_save_path = os.path.join(model_dir, "best_student_model.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"Saved the best model to {model_save_path}")
    
    # Also save the features info for validation
    info_save_path = os.path.join(model_dir, "model_features.joblib")
    joblib.dump(features, info_save_path)
    
    # Append results to the markdown report
    append_model_results_to_report(results, best_model_name, best_r2)

def append_model_results_to_report(results, best_model, best_r2, report_path="reports/performance_report.md"):
    """Appends machine learning training results to the performance report."""
    if not os.path.exists(report_path):
        return
        
    with open(report_path, "a") as f:
        f.write("\n## 9. Machine Learning Predictive Analysis\n")
        f.write("We trained three regression models to predict final exam marks based on attendance, study hours, sleep hours, and previous grades.\n\n")
        f.write("| Model Name | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-squared (R2) Score |\n")
        f.write("|---|---|---|---|\n")
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | {metrics['R2']:.4f} |\n")
        f.write("\n")
        f.write(f"**Conclusion**: The **{best_model}** model outperformed the other models with a R-squared score of **{best_r2:.4%}**, and has been exported for integration into the interactive dashboard.\n")
        
    print("Appended model results to report.")

if __name__ == "__main__":
    train_and_evaluate_models()
