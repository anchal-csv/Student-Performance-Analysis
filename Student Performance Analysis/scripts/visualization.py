import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for modern premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8f9fa",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif"
})

# Palette colors
primary_color = "#1f77b4" # Soft blue
accent_color = "#ff7f0e"  # Soft orange
success_color = "#2ca02c" # Soft green
danger_color = "#d62728"  # Soft red

def generate_visualizations(df, output_dir="reports/charts"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating charts and saving to {output_dir}...")
    
    # 1. Distribution of Marks
    plt.figure(figsize=(9, 5))
    sns.histplot(df["Final_Exam_Marks"], kde=True, color=primary_color, bins=25, edgecolor="#ffffff", linewidth=1.2)
    plt.axvline(df["Final_Exam_Marks"].mean(), color=danger_color, linestyle="--", linewidth=1.5, label=f"Average: {df['Final_Exam_Marks'].mean():.1f}")
    plt.axvline(50, color=success_color, linestyle="-", linewidth=1.5, label="Pass Mark (50)")
    plt.title("Distribution of Student Final Exam Marks", pad=15)
    plt.xlabel("Final Exam Marks (Out of 100)")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/marks_distribution.png", dpi=200)
    plt.close()
    
    # 2. Attendance vs Marks Scatter Plot with Regression Line
    plt.figure(figsize=(9, 5.5))
    sns.regplot(data=df, x="Attendance_Percentage", y="Final_Exam_Marks", 
                scatter_kws={"alpha": 0.5, "color": primary_color, "edgecolor": "#ffffff", "s": 35},
                line_kws={"color": danger_color, "linewidth": 2, "label": "Regression Trend"})
    plt.title("Impact of Attendance on Final Exam Marks", pad=15)
    plt.xlabel("Attendance Percentage (%)")
    plt.ylabel("Final Exam Marks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attendance_vs_marks.png", dpi=200)
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 7))
    numeric_cols = ["Attendance_Percentage", "Study_Hours", "Sleep_Hours", "Internet_Usage_Hours", "Previous_Grades", "Final_Exam_Marks"]
    corr_matrix = df[numeric_cols].corr()
    
    # Rename columns for cleaner heatmap display
    cols_display = ["Attendance (%)", "Study Hours", "Sleep Hours", "Internet Hours", "Prev Grades", "Final Exam Marks"]
    corr_matrix.columns = cols_display
    corr_matrix.index = cols_display
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap="coolwarm", fmt=".2f", 
                linewidths=0.5, vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix Heatmap", pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=200)
    plt.close()
    
    # 4. Pass vs Fail Ratio
    plt.figure(figsize=(6, 6))
    counts = df["Result"].value_counts()
    colors = [success_color, danger_color][:len(counts)]
    explode = [0.05] * len(counts)
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140, 
            colors=colors, explode=explode, textprops={'fontsize': 12}, 
            wedgeprops={'edgecolor': '#ffffff', 'linewidth': 1.5, 'antialiased': True})
    plt.title("Pass vs Fail Ratio", pad=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pass_fail_ratio.png", dpi=200)
    plt.close()
    
    # 5. Study Hours Impact (Average Marks by Category)
    study_bins = [0, 3, 6, 11]
    study_labels = ["Low (<3 hrs)", "Medium (3-6 hrs)", "High (>6 hrs)"]
    df_temp = df.copy()
    df_temp["Study_Category"] = pd.cut(df_temp["Study_Hours"], bins=study_bins, labels=study_labels, right=False)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_temp, x="Study_Category", y="Final_Exam_Marks", errorbar=None, palette="Blues_d", hue="Study_Category", legend=False)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha="center", va="bottom", fontsize=11, color="black", 
                    xytext=(0, 5), textcoords="offset points")
                    
    plt.title("Average Final Exam Marks by Study Hours Category", pad=15)
    plt.xlabel("Daily Study Hours")
    plt.ylabel("Average Final Exam Marks")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/study_hours_vs_marks.png", dpi=200)
    plt.close()
    
    # 6. Sleep Hours Impact
    sleep_bins = [0, 6, 8.5, 11]
    sleep_labels = ["Insufficient (<6 hrs)", "Healthy (6-8.5 hrs)", "Excessive (>8.5 hrs)"]
    df_temp["Sleep_Category"] = pd.cut(df_temp["Sleep_Hours"], bins=sleep_bins, labels=sleep_labels, right=False)
    
    plt.figure(figsize=(8, 5.5))
    sns.boxplot(data=df_temp, x="Sleep_Category", y="Final_Exam_Marks", palette="Set2", hue="Sleep_Category", legend=False)
    plt.title("Effect of Sleep on Academic Results", pad=15)
    plt.xlabel("Sleep Hours Category")
    plt.ylabel("Final Exam Marks Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sleep_vs_marks.png", dpi=200)
    plt.close()

    # 7. Performance by Gender
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=df_temp, x="Gender", y="Final_Exam_Marks", errorbar=None, palette=["#3498db", "#e74c3c"], hue="Gender", legend=False)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha="center", va="bottom", fontsize=11, color="black", 
                    xytext=(0, 5), textcoords="offset points")
                    
    plt.title("Performance Comparison by Gender", pad=15)
    plt.ylabel("Average Final Exam Marks")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gender_performance.png", dpi=200)
    plt.close()
    
    # 8. Performance by Department
    plt.figure(figsize=(9, 5))
    dept_order = df.groupby("Department")["Final_Exam_Marks"].mean().sort_values(ascending=False).index
    ax = sns.barplot(data=df, y="Department", x="Final_Exam_Marks", errorbar=None, 
                     palette="viridis", order=dept_order, hue="Department", legend=False)
    
    # Add values on end of bars
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.1f}", 
                    (width, p.get_y() + p.get_height() / 2.), 
                    ha="left", va="center", fontsize=11, color="black", 
                    xytext=(5, 0), textcoords="offset points")
                    
    plt.title("Average Final Exam Marks by Department", pad=15)
    plt.xlabel("Average Final Exam Marks")
    plt.ylabel("Department")
    plt.xlim(0, 110)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/department_performance.png", dpi=200)
    plt.close()
    
    print("All charts generated and saved successfully!")

if __name__ == "__main__":
    cleaned_csv = "dataset/student_performance_cleaned.csv"
    if not os.path.exists(cleaned_csv):
        print(f"Cleaned dataset not found at {cleaned_csv}. Run data_cleaning.py first.")
    else:
        df = pd.read_csv(cleaned_csv)
        generate_visualizations(df)
