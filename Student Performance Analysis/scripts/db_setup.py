import sqlite3
import pandas as pd
import os

db_path = "dataset/student_database.db"
csv_path = "dataset/student_performance.csv"

# Remove old db if exists to start fresh
if os.path.exists(db_path):
    os.remove(db_path)

print(f"Creating database at {db_path}...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table matching dataset columns
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Student_ID TEXT,
    Name TEXT,
    Gender TEXT,
    Age INTEGER,
    Department TEXT,
    Attendance_Percentage REAL,
    Study_Hours REAL,
    Sleep_Hours REAL,
    Internet_Usage_Hours REAL,
    Previous_Grades REAL,
    Family_Income TEXT,
    Parent_Education TEXT,
    Final_Exam_Marks REAL,
    Result TEXT
)
""")

# Load CSV and insert into database
df = pd.read_csv(csv_path)
# Replace NaN with None so they are inserted as NULL in SQLite
df_db = df.where(pd.notnull(df), None)

print(f"Inserting {len(df_db)} records into database...")
for idx, row in df_db.iterrows():
    cursor.execute("""
    INSERT INTO students (
        Student_ID, Name, Gender, Age, Department, Attendance_Percentage,
        Study_Hours, Sleep_Hours, Internet_Usage_Hours, Previous_Grades,
        Family_Income, Parent_Education, Final_Exam_Marks, Result
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["Student_ID"], row["Name"], row["Gender"], int(row["Age"]) if row["Age"] is not None else None,
        row["Department"], row["Attendance_Percentage"], row["Study_Hours"], row["Sleep_Hours"],
        row["Internet_Usage_Hours"], row["Previous_Grades"], row["Family_Income"], row["Parent_Education"],
        row["Final_Exam_Marks"], row["Result"]
    ))

conn.commit()
print("Database population completed successfully!")

# Verify row count
cursor.execute("SELECT COUNT(*) FROM students")
count = cursor.fetchone()[0]
print(f"Total rows in SQLite database table 'students': {count}")

conn.close()
