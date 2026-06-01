import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Create dataset directory if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Define dataset size
num_students = 1000

# Lists of options for categorical variables
first_names_m = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", 
                 "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua"]
first_names_f = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
                 "Lisa", "Nancy", "Betty", "Sandra", "Margaret", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

departments = ["Computer Science", "Information Technology", "Mechanical Engineering", "Electrical Engineering", "Business Administration"]
family_incomes = ["Low", "Medium", "High"]
parent_educations = ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD"]

# Generate data
student_ids = [f"STU{i:04d}" for i in range(1, num_students + 1)]
genders = np.random.choice(["Male", "Female"], size=num_students, p=[0.48, 0.52])

names = []
for g in genders:
    fname = np.random.choice(first_names_m) if g == "Male" else np.random.choice(first_names_f)
    lname = np.random.choice(last_names)
    names.append(f"{fname} {lname}")

ages = np.random.randint(18, 23, size=num_students)
departments_chosen = np.random.choice(departments, size=num_students)

# Attendance percentage with a realistic distribution (mean=82, truncated at 100, min around 50)
attendance = np.clip(np.random.normal(loc=82, scale=10, size=num_students), 50, 100).round(1)

# Study hours per day (mean=5, range 1-10)
study_hours = np.clip(np.random.normal(loc=5, scale=2.2, size=num_students), 1, 10).round(1)

# Sleep hours per night (mean=7, range 4-10)
sleep_hours = np.clip(np.random.normal(loc=7, scale=1.2, size=num_students), 4, 10).round(1)

# Internet usage hours per day (mean=3.5, range 1-8)
internet_usage = np.clip(np.random.normal(loc=3.5, scale=1.5, size=num_students), 1, 8).round(1)

# Previous grades (out of 100, mean=70, scale=12, correlated with study hours/attendance)
base_prev_grades = np.random.normal(loc=65, scale=10, size=num_students)
prev_grades = np.clip(base_prev_grades + 1.2 * study_hours + 0.15 * attendance, 40, 100).round(1)

family_income_chosen = np.random.choice(family_incomes, size=num_students, p=[0.3, 0.5, 0.2])
parent_edu_chosen = np.random.choice(parent_educations, size=num_students, p=[0.25, 0.2, 0.35, 0.15, 0.05])

# Calculate final exam marks based on factors, plus a random error
# Higher study hours, attendance, sleep (up to a point), and previous grades help.
# Higher internet usage might have a small negative effect.
# Let's map parent education and family income to slight boosts (representing resources).
parent_edu_map = {"High School": 0, "Associate's Degree": 2, "Bachelor's Degree": 4, "Master's Degree": 6, "PhD": 8}
family_income_map = {"Low": 0, "Medium": 3, "High": 6}

# Sleep sweet spot is 7.5 hours. Penalty for deviance.
sleep_dev = np.abs(sleep_hours - 7.5)
sleep_factor = 5 - 2.5 * sleep_dev

# Final Exam Marks formula
final_marks = (
    0.25 * attendance +
    3.5 * study_hours +
    sleep_factor +
    0.35 * prev_grades -
    0.5 * internet_usage +
    np.array([parent_edu_map[pe] for pe in parent_edu_chosen]) +
    np.array([family_income_map[fi] for fi in family_income_chosen]) +
    np.random.normal(loc=-15, scale=5, size=num_students) # intercept & noise
)

# Clip final exam marks to 0-100 range
final_exam_marks = np.clip(final_marks, 0, 100).round(1)

# Result (Pass/Fail) - Pass mark is 50
results = np.where(final_exam_marks >= 50, "Pass", "Fail")

# Create DataFrame
df = pd.DataFrame({
    "Student_ID": student_ids,
    "Name": names,
    "Gender": genders,
    "Age": ages,
    "Department": departments_chosen,
    "Attendance_Percentage": attendance,
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Internet_Usage_Hours": internet_usage,
    "Previous_Grades": prev_grades,
    "Family_Income": family_income_chosen,
    "Parent_Education": parent_edu_chosen,
    "Final_Exam_Marks": final_exam_marks,
    "Result": results
})

# Let's introduce a tiny amount of missing values (e.g. 1% missingness in Sleep_Hours and Family_Income)
# to demonstrate that our data_cleaning.py script actually cleans and handles missing values.
missing_sleep_idx = np.random.choice(num_students, size=10, replace=False)
missing_income_idx = np.random.choice(num_students, size=15, replace=False)
df.loc[missing_sleep_idx, "Sleep_Hours"] = np.nan
df.loc[missing_income_idx, "Family_Income"] = None

# Let's add 5 duplicate rows to demonstrate deduplication
duplicates = df.sample(n=5, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# Save to CSV
output_path = "dataset/student_performance.csv"
df.to_csv(output_path, index=False)
print(f"Dataset generated successfully at {output_path}. Shape: {df.shape}")
