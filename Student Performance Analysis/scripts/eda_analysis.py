import os
import pandas as pd
import numpy as np

def run_eda(df):
    """
    Performs exploratory data analysis on the cleaned student dataset.
    Returns a dictionary of structured statistics.
    """
    stats = {}
    
    # 1. Basic Metrics
    stats["total_students"] = len(df)
    stats["avg_attendance"] = df["Attendance_Percentage"].mean()
    stats["avg_study_hours"] = df["Study_Hours"].mean()
    stats["avg_sleep_hours"] = df["Sleep_Hours"].mean()
    stats["avg_previous_grades"] = df["Previous_Grades"].mean()
    stats["avg_final_marks"] = df["Final_Exam_Marks"].mean()
    
    # 2. Pass vs Fail Ratio
    pass_fail_counts = df["Result"].value_counts()
    stats["pass_count"] = pass_fail_counts.get("Pass", 0)
    stats["fail_count"] = pass_fail_counts.get("Fail", 0)
    stats["pass_rate"] = (stats["pass_count"] / stats["total_students"]) * 100
    
    # 3. Average Marks by Study Hours Category
    # Define categories: Low (<3 hrs), Medium (3-6 hrs), High (>6 hrs)
    study_bins = [0, 3, 6, 11]
    study_labels = ["Low (<3 hrs)", "Medium (3-6 hrs)", "High (>6 hrs)"]
    df["Study_Hours_Category"] = pd.cut(df["Study_Hours"], bins=study_bins, labels=study_labels, right=False)
    stats["marks_by_study_hours"] = df.groupby("Study_Hours_Category", observed=False)["Final_Exam_Marks"].agg(["mean", "count"]).to_dict("index")
    
    # 4. Attendance Impact on Performance
    # Categories: Poor (<75%), Average (75-90%), Excellent (90-100%)
    att_bins = [0, 75, 90, 101]
    att_labels = ["Poor (<75%)", "Average (75-90%)", "Excellent (90-100%)"]
    df["Attendance_Category"] = pd.cut(df["Attendance_Percentage"], bins=att_bins, labels=att_labels, right=False)
    stats["marks_by_attendance"] = df.groupby("Attendance_Category", observed=False)["Final_Exam_Marks"].agg(["mean", "count"]).to_dict("index")
    
    # 5. Correlation Heatmap Data (numeric fields only)
    numeric_cols = ["Attendance_Percentage", "Study_Hours", "Sleep_Hours", "Internet_Usage_Hours", "Previous_Grades", "Final_Exam_Marks"]
    stats["correlation_matrix"] = df[numeric_cols].corr().to_dict()
    
    # 6. Top 10 and Bottom 10 Students
    top_10 = df.sort_values(by="Final_Exam_Marks", ascending=False).head(10)[["Student_ID", "Name", "Gender", "Department", "Attendance_Percentage", "Study_Hours", "Final_Exam_Marks"]]
    stats["top_10_students"] = top_10.to_dict("records")
    
    bottom_10 = df.sort_values(by="Final_Exam_Marks", ascending=True).head(10)[["Student_ID", "Name", "Gender", "Department", "Attendance_Percentage", "Study_Hours", "Final_Exam_Marks"]]
    stats["bottom_10_students"] = bottom_10.to_dict("records")
    
    # 7. Gender Comparison
    stats["marks_by_gender"] = df.groupby("Gender")["Final_Exam_Marks"].mean().to_dict()
    # Pass rate by gender
    gender_pass_df = df.groupby("Gender")["Result"].value_counts(normalize=True).unstack() * 100
    stats["pass_rate_by_gender"] = gender_pass_df.to_dict()
    
    # 8. Department Performance Comparison
    stats["marks_by_department"] = df.groupby("Department")["Final_Exam_Marks"].mean().to_dict()
    dept_pass_df = df.groupby("Department")["Result"].value_counts(normalize=True).unstack() * 100
    stats["pass_rate_by_department"] = dept_pass_df.to_dict()
    
    # 9. Sleep Impact
    # Categories: Insufficient (<6 hrs), Healthy (6-8 hrs), Excessive (>8 hrs)
    sleep_bins = [0, 6, 8.5, 11]
    sleep_labels = ["Insufficient (<6 hrs)", "Healthy (6-8.5 hrs)", "Excessive (>8.5 hrs)"]
    df["Sleep_Category"] = pd.cut(df["Sleep_Hours"], bins=sleep_bins, labels=sleep_labels, right=False)
    stats["marks_by_sleep"] = df.groupby("Sleep_Category", observed=False)["Final_Exam_Marks"].agg(["mean", "count"]).to_dict("index")
    
    return stats

def generate_markdown_report(stats, output_path="reports/performance_report.md"):
    """Generates a detailed markdown report detailing EDA findings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Student Performance Analysis Report\n\n")
        f.write("This report presents exploratory data analysis and key insights from the student performance dataset.\n\n")
        
        # Summary metrics
        f.write("## 1. Key Performance Indicators (KPIs)\n")
        f.write(f"- **Total Students Analyzed**: {stats['total_students']}\n")
        f.write(f"- **Average Attendance**: {stats['avg_attendance']:.2f}%\n")
        f.write(f"- **Average Study Hours/Day**: {stats['avg_study_hours']:.2f} hrs\n")
        f.write(f"- **Average Sleep Hours/Night**: {stats['avg_sleep_hours']:.2f} hrs\n")
        f.write(f"- **Average Previous Grade**: {stats['avg_previous_grades']:.2f}/100\n")
        f.write(f"- **Average Final Exam Marks**: {stats['avg_final_marks']:.2f}/100\n")
        f.write(f"- **Overall Pass Rate**: {stats['pass_rate']:.2f}% ({stats['pass_count']} passed, {stats['fail_count']} failed)\n\n")
        
        # Study Hours comparison
        f.write("## 2. Impact of Study Hours on Final Marks\n")
        f.write("| Study Category | Average Final Marks | Student Count |\n")
        f.write("|---|---|---|\n")
        for cat, values in stats["marks_by_study_hours"].items():
            f.write(f"| {cat} | {values['mean']:.2f} | {values['count']} |\n")
        f.write("\n")
        
        # Attendance Impact
        f.write("## 3. Impact of Attendance on Final Marks\n")
        f.write("| Attendance Category | Average Final Marks | Student Count |\n")
        f.write("|---|---|---|\n")
        for cat, values in stats["marks_by_attendance"].items():
            f.write(f"| {cat} | {values['mean']:.2f} | {values['count']} |\n")
        f.write("\n")
        
        # Sleep Impact
        f.write("## 4. Impact of Sleep on Academic Results\n")
        f.write("| Sleep Category | Average Final Marks | Student Count |\n")
        f.write("|---|---|---|\n")
        for cat, values in stats["marks_by_sleep"].items():
            f.write(f"| {cat} | {values['mean']:.2f} | {values['count']} |\n")
        f.write("\n")
        
        # Gender & Department Comparison
        f.write("## 5. Demographic and Department Comparison\n")
        f.write("### Performance by Gender\n")
        f.write("| Gender | Average Marks | Pass Rate (%) |\n")
        f.write("|---|---|---|\n")
        for gender in stats["marks_by_gender"].keys():
            pass_rate = stats["pass_rate_by_gender"]["Pass"].get(gender, 0)
            f.write(f"| {gender} | {stats['marks_by_gender'][gender]:.2f} | {pass_rate:.2f}% |\n")
        f.write("\n")
        
        f.write("### Performance by Department\n")
        f.write("| Department | Average Marks | Pass Rate (%) |\n")
        f.write("|---|---|---|\n")
        for dept in stats["marks_by_department"].keys():
            pass_rate = stats["pass_rate_by_department"]["Pass"].get(dept, 0)
            f.write(f"| {dept} | {stats['marks_by_department'][dept]:.2f} | {pass_rate:.2f}% |\n")
        f.write("\n")
        
        # Correlation Matrix
        f.write("## 6. Correlation Analysis\n")
        f.write("Correlation values with Final_Exam_Marks:\n")
        final_marks_corr = stats["correlation_matrix"]["Final_Exam_Marks"]
        f.write("| Variable | Correlation with Final Marks |\n")
        f.write("|---|---|\n")
        for var, corr in sorted(final_marks_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            if var != "Final_Exam_Marks":
                f.write(f"| {var} | {corr:.4f} |\n")
        f.write("\n")
        
        # Top 10 Students
        f.write("## 7. Top 10 Performing Students\n")
        f.write("| Student ID | Name | Department | Attendance | Study Hours | Final Marks |\n")
        f.write("|---|---|---|---|---|---|\n")
        for student in stats["top_10_students"]:
            f.write(f"| {student['Student_ID']} | {student['Name']} | {student['Department']} | {student['Attendance_Percentage']}% | {student['Study_Hours']} | {student['Final_Exam_Marks']} |\n")
        f.write("\n")
        
        # Bottom 10 Students
        f.write("## 8. Bottom 10 Performing Students\n")
        f.write("| Student ID | Name | Department | Attendance | Study Hours | Final Marks |\n")
        f.write("|---|---|---|---|---|---|\n")
        for student in stats["bottom_10_students"]:
            f.write(f"| {student['Student_ID']} | {student['Name']} | {student['Department']} | {student['Attendance_Percentage']}% | {student['Study_Hours']} | {student['Final_Exam_Marks']} |\n")
        f.write("\n")
        
    print(f"Markdown report generated successfully at {output_path}")

if __name__ == "__main__":
    cleaned_csv = "dataset/student_performance_cleaned.csv"
    if not os.path.exists(cleaned_csv):
        print(f"Cleaned dataset not found at {cleaned_csv}. Run data_cleaning.py first.")
    else:
        df = pd.read_csv(cleaned_csv)
        stats = run_eda(df)
        generate_markdown_report(stats)
