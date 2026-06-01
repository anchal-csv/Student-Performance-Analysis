
#  Student Academic Performance Analytics System

A comprehensive, production-grade data science project designed to analyze, visualize, and predict student academic performance. The system integrates a SQLite database, automated data cleaning pipelines, exploratory data analysis (EDA), predictive machine learning modeling, and an interactive Streamlit analytics dashboard.

---

##  Project Architecture

```
Student Performance Analysis/
│
├── dashboard/                      # Dashboard components
│   ├── dashboard.py                # Main Streamlit dashboard application
│   ├── best_student_model.joblib   # Serialized best ML model (Linear Regression)
│   └── model_features.joblib       # Serialized feature list for validation
│
├── dataset/                        # Data storage
│   ├── student_database.db         # SQLite database storing raw student records
│   ├── student_performance.csv     # Raw dataset generated dynamically
│   └── student_performance_cleaned.csv # Cleaned & preprocessed dataset
│
├── reports/                        # Analysis reports & visualization outputs
│   ├── performance_report.md       # Comprehensive markdown performance report
│   └── charts/                     # Exported Matplotlib/Seaborn visualization figures
│       ├── attendance_vs_marks.png
│       ├── correlation_heatmap.png
│       ├── department_performance.png
│       ├── gender_performance.png
│       ├── marks_distribution.png
│       ├── pass_fail_ratio.png
│       ├── sleep_vs_marks.png
│       └── study_hours_vs_marks.png
│
├── scripts/                        # Data processing & modeling pipeline scripts
│   ├── generate_dataset.py         # Dynamic synthetic student dataset generator
│   ├── db_setup.py                 # SQLite database initializer and population script
│   ├── data_cleaning.py            # Deduplication, missing value imputation, type casting
│   ├── eda_analysis.py             # Statistical calculations & markdown report exporter
│   ├── visualization.py            # Seaborn/Matplotlib chart generator
│   └── prediction_model.py         # ML training (Linear Regression, Decision Tree, Random Forest)
│
├── requirements.txt                # Project dependencies
└── README.md                       # System documentation (This file)
```

---

##  Quick Start & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Setup Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the package manager:
```bash
pip install -r requirements.txt
```

---

##  Data Pipeline Execution

You can run each stage of the pipeline independently in the following order:

```bash
# 1. Generate the synthetic student dataset
python scripts/generate_dataset.py

# 2. Set up and populate the SQLite Database
python scripts/db_setup.py

# 3. Load, clean, and preprocess the dataset
python scripts/data_cleaning.py

# 4. Generate statistical reports
python scripts/eda_analysis.py

# 5. Export static Seaborn/Matplotlib visualization charts
python scripts/visualization.py

# 6. Train, evaluate machine learning models, and save the best one
python scripts/prediction_model.py
```

---

##  Launching the Streamlit Dashboard

Start the premium interactive dashboard with Streamlit:
```bash
streamlit run dashboard/dashboard.py
```

Once running, open your browser and navigate to `http://localhost:8501`.

---

##  System Features

### 1. Streamlit Dashboard tabs:
* **Performance Overview**: Features dynamic Plotly charts detailing final mark distributions, pass/fail ratios, average marks by department, and gender-based boxplots.
* **Factors & Correlations**: Scatter plots with regression lines tracking how study hours and attendance percentages affect scores, sleep analysis boxplots, and an interactive correlation matrix heatmap.
* **Top & Low Performers**: Simple record search system by Name or Student ID, displaying top 10 and bottom 10 academic performers.
* **ML Mark Predictor**: Allows users to input a hypothetical student's details (attendance, sleep, study hours, previous grades) and view their predicted score using a gauge chart and customized academic recommendations.
* **SQL Sandbox & Console**: A read-only console to write and execute SQL SELECT queries directly against the live SQLite database table `students`.

###  2. Machine Learning Predictive Analysis
We train and evaluate three different regression models to predict final exam scores based on student habits:
* **Linear Regression (Best Model - Selected)**: R² Score: **67.63%** | MAE: **4.64**
* **Random Forest Regressor**: R² Score: **66.06%** | MAE: **4.62**
* **Decision Tree Regressor**: R² Score: **58.44%** | MAE: **5.09**

---

## Key Insights from Data Analysis
* **Study Hours**: Shows the highest positive correlation (+0.716) with final marks. Students studying >6 hours/day average **67.54** marks, compared to **47.46** marks for students studying <3 hours.
* **Attendance**: Attendance percentage shows a strong positive correlation (+0.231). Excellent attendance (>90%) yields average marks of **61.23**.
* **Sleep**: Academic performance peaks when sleep is in the healthy range (6–8.5 hours), averaging **59.46** marks. Cognitive retention drops with sleep durations <6 hours.
