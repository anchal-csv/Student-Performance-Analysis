import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium dashboard design (Dark theme accents, glassmorphism card styling)
st.markdown("""
<style>
    /* Main container and text */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header card styling */
    .header-box {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .header-box h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* Custom KPI Cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background-color: #1a1c23;
        border: 1px solid #2d313f;
        padding: 1.5rem;
        border-radius: 10px;
        flex: 1;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #00adb5;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00adb5;
        margin-bottom: 0.2rem;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #8f9bb3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Recommendations Box */
    .rec-box {
        background-color: #162447;
        border-left: 5px solid #00adb5;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to connect to SQLite
def query_db(sql_query, db_path="dataset/student_database.db"):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()

# Load cleaned dataset
@st.cache_data
def load_cleaned_data(path="dataset/student_performance_cleaned.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback to DB if cleaned CSV doesn't exist
    conn = sqlite3.connect("dataset/student_database.db")
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    # Basic cleaning on fly
    df_cleaned = df.drop_duplicates(subset=["Student_ID"])
    df_cleaned["Sleep_Hours"] = df_cleaned["Sleep_Hours"].fillna(df_cleaned["Sleep_Hours"].median())
    df_cleaned["Family_Income"] = df_cleaned["Family_Income"].fillna(df_cleaned["Family_Income"].mode()[0])
    return df_cleaned

# Load Machine Learning Model
@st.cache_resource
def load_ml_model(model_path="dashboard/best_student_model.joblib"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df_data = load_cleaned_data()
model = load_ml_model()

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://img.icons8.com/color/96/000000/student-male--v1.png", width=80)
st.sidebar.title("Data Filters")

# Gender Filter
all_genders = sorted(df_data["Gender"].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", all_genders, default=all_genders)

# Department Filter
all_depts = sorted(df_data["Department"].unique().tolist())
selected_depts = st.sidebar.multiselect("Department", all_depts, default=all_depts)

# Family Income Filter
all_incomes = sorted(df_data["Family_Income"].unique().tolist())
selected_incomes = st.sidebar.multiselect("Family Income Level", all_incomes, default=all_incomes)

# Attendance Slider
min_att = float(df_data["Attendance_Percentage"].min())
max_att = float(df_data["Attendance_Percentage"].max())
selected_att = st.sidebar.slider("Attendance Percentage Range", min_att, max_att, (min_att, max_att))

# Study Hours Slider
min_study = float(df_data["Study_Hours"].min())
max_study = float(df_data["Study_Hours"].max())
selected_study = st.sidebar.slider("Daily Study Hours Range", min_study, max_study, (min_study, max_study))

# Filter dataframe based on selections
filtered_df = df_data[
    (df_data["Gender"].isin(selected_genders)) &
    (df_data["Department"].isin(selected_depts)) &
    (df_data["Family_Income"].isin(selected_incomes)) &
    (df_data["Attendance_Percentage"] >= selected_att[0]) &
    (df_data["Attendance_Percentage"] <= selected_att[1]) &
    (df_data["Study_Hours"] >= selected_study[0]) &
    (df_data["Study_Hours"] <= selected_study[1])
]

# --- MAIN PAGE HEADER ---
st.markdown("""
<div class="header-box">
    <h1>🎓 Student Academic Performance Analytics</h1>
    <p>Perform exploratory data analysis, analyze performance correlations, run custom database queries, and predict student exam results using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("No data match the selected filters. Please adjust your sidebar settings.")
else:
    # --- KPI CARDS ROW ---
    total_students = len(filtered_df)
    avg_attendance = filtered_df["Attendance_Percentage"].mean()
    avg_study = filtered_df["Study_Hours"].mean()
    avg_sleep = filtered_df["Sleep_Hours"].mean()
    avg_marks = filtered_df["Final_Exam_Marks"].mean()
    
    pass_count = (filtered_df["Result"] == "Pass").sum()
    pass_rate = (pass_count / total_students) * 100 if total_students > 0 else 0
    
    kpi_cols = st.columns(6)
    
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_students}</div>
            <div class="kpi-label">Total Students</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_attendance:.1f}%</div>
            <div class="kpi-label">Avg Attendance</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_study:.1f} hrs</div>
            <div class="kpi-label">Avg Study Hours</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[3]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_sleep:.1f} hrs</div>
            <div class="kpi-label">Avg Sleep Hours</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[4]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_marks:.1f}</div>
            <div class="kpi-label">Avg Exam Marks</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[5]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{pass_rate:.1f}%</div>
            <div class="kpi-label">Pass Rate</div>
        </div>
        """, unsafe_allow_html=True)

    # --- TABS FOR ANALYSIS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Overview", 
        "🔍 Factors & Correlations", 
        "🏆 Top & Low Performers", 
        "🧠 ML Mark Predictor", 
        "💻 SQL Console & Data Export"
    ])

    # --- TAB 1: PERFORMANCE OVERVIEW ---
    with tab1:
        col1_1, col1_2 = st.columns([2, 1])
        
        with col1_1:
            st.subheader("Distribution of Final Exam Marks")
            fig_hist = px.histogram(
                filtered_df, 
                x="Final_Exam_Marks", 
                nbins=20, 
                color_discrete_sequence=["#00adb5"], 
                marginal="box",
                labels={"Final_Exam_Marks": "Final Exam Marks"}
            )
            fig_hist.add_vline(x=50, line_dash="dash", line_color="#d62728", annotation_text="Pass Mark (50)", annotation_position="top left")
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col1_2:
            st.subheader("Pass vs Fail Ratio")
            fig_pie = px.pie(
                filtered_df, 
                names="Result", 
                color="Result",
                color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"},
                hole=0.4
            )
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        col1_3, col1_4 = st.columns(2)
        
        with col1_3:
            st.subheader("Average Final Marks by Department")
            dept_avg = filtered_df.groupby("Department")["Final_Exam_Marks"].mean().reset_index().sort_values(by="Final_Exam_Marks", ascending=False)
            fig_dept = px.bar(
                dept_avg, 
                x="Final_Exam_Marks", 
                y="Department", 
                orientation="h",
                color="Final_Exam_Marks",
                color_continuous_scale="Viridis",
                labels={"Final_Exam_Marks": "Average Marks"}
            )
            fig_dept.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_dept, use_container_width=True)
            
        with col1_4:
            st.subheader("Gender Performance Analysis")
            fig_gender = px.box(
                filtered_df, 
                x="Gender", 
                y="Final_Exam_Marks", 
                color="Gender",
                color_discrete_sequence=["#3498db", "#e74c3c"]
            )
            fig_gender.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_gender, use_container_width=True)

    # --- TAB 2: FACTORS & CORRELATIONS ---
    with tab2:
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.subheader("Attendance vs Final Exam Marks")
            fig_att_scatter = px.scatter(
                filtered_df, 
                x="Attendance_Percentage", 
                y="Final_Exam_Marks", 
                color="Result",
                hover_data=["Name", "Department", "Study_Hours"],
                color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"},
                trendline="ols",
                trendline_color_override="#ff7f0e"
            )
            fig_att_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_att_scatter, use_container_width=True)
            
        with col2_2:
            st.subheader("Study Hours vs Final Exam Marks")
            fig_study_scatter = px.scatter(
                filtered_df, 
                x="Study_Hours", 
                y="Final_Exam_Marks", 
                color="Department",
                hover_data=["Name", "Attendance_Percentage"],
                trendline="ols"
            )
            fig_study_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_study_scatter, use_container_width=True)
            
        col2_3, col2_4 = st.columns(2)
        
        with col2_3:
            st.subheader("Effect of Sleep on Academic Results")
            # Group sleep hours into bins for visual analysis
            sleep_bins = [0, 6, 8.5, 11]
            sleep_labels = ["Insufficient (<6 hrs)", "Healthy (6-8.5 hrs)", "Excessive (>8.5 hrs)"]
            df_sleep = filtered_df.copy()
            df_sleep["Sleep_Category"] = pd.cut(df_sleep["Sleep_Hours"], bins=sleep_bins, labels=sleep_labels, right=False)
            
            fig_sleep = px.box(
                df_sleep, 
                x="Sleep_Category", 
                y="Final_Exam_Marks", 
                color="Sleep_Category",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_sleep.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            st.plotly_chart(fig_sleep, use_container_width=True)
            
        with col2_4:
            st.subheader("Correlation Matrix Heatmap")
            numeric_cols = ["Attendance_Percentage", "Study_Hours", "Sleep_Hours", "Internet_Usage_Hours", "Previous_Grades", "Final_Exam_Marks"]
            corr = filtered_df[numeric_cols].corr()
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=["Attendance", "Study Hours", "Sleep Hours", "Internet Hours", "Prev Grades", "Final Marks"],
                y=["Attendance", "Study Hours", "Sleep Hours", "Internet Hours", "Prev Grades", "Final Marks"],
                colorscale="RdBu",
                zmin=-1, zmax=1,
                text=np.around(corr.values, 2),
                texttemplate="%{text}",
                showscale=True
            ))
            fig_heat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)", 
                font_color="#e0e6ed",
                height=400
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # --- TAB 3: TOP & LOW PERFORMERS ---
    with tab3:
        st.subheader("Search Student Records")
        search_query = st.text_input("Search student by name or ID:")
        
        display_df = filtered_df.copy()
        if search_query:
            display_df = display_df[
                display_df["Name"].str.contains(search_query, case=False, na=False) |
                display_df["Student_ID"].str.contains(search_query, case=False, na=False)
            ]
            
        st.dataframe(display_df, use_container_width=True)
        
        col3_1, col3_2 = st.columns(2)
        
        with col3_1:
            st.subheader("🏆 Top 10 Performing Students")
            top_10 = filtered_df.sort_values(by="Final_Exam_Marks", ascending=False).head(10)[
                ["Student_ID", "Name", "Department", "Attendance_Percentage", "Study_Hours", "Final_Exam_Marks"]
            ]
            st.dataframe(top_10, use_container_width=True)
            
        with col3_2:
            st.subheader("⚠️ Low 10 Performing Students")
            bottom_10 = filtered_df.sort_values(by="Final_Exam_Marks", ascending=True).head(10)[
                ["Student_ID", "Name", "Department", "Attendance_Percentage", "Study_Hours", "Final_Exam_Marks"]
            ]
            st.dataframe(bottom_10, use_container_width=True)

    # --- TAB 4: ML MARK PREDICTOR ---
    with tab4:
        st.subheader("Predict Student Final Exam Marks using Machine Learning")
        
        if model is None:
            st.error("Prediction Model file not found. Please train models first by running prediction_model.py")
        else:
            st.write("Enter the student academic metrics below to estimate their final exam marks:")
            
            col4_1, col4_2 = st.columns(2)
            
            with col4_1:
                input_attendance = st.slider("Attendance Percentage (%)", 50.0, 100.0, 85.0, step=0.5)
                input_study = st.slider("Daily Study Hours", 1.0, 10.0, 5.0, step=0.1)
                
            with col4_2:
                input_sleep = st.slider("Daily Sleep Hours", 4.0, 10.0, 7.0, step=0.1)
                input_prev_grades = st.slider("Previous Grades (out of 100)", 40.0, 100.0, 75.0, step=0.5)
                
            # Perform prediction
            input_features = pd.DataFrame([{
                'Attendance_Percentage': input_attendance,
                'Study_Hours': input_study,
                'Sleep_Hours': input_sleep,
                'Previous_Grades': input_prev_grades
            }])
            
            predicted_mark = model.predict(input_features)[0]
            predicted_mark = np.clip(predicted_mark, 0.0, 100.0)
            predicted_result = "Pass" if predicted_mark >= 50.0 else "Fail"
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                # Gauge plot for predicted score
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_mark,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Exam Mark", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#00adb5"},
                        'bgcolor': "#1a1c23",
                        'borderwidth': 2,
                        'bordercolor': "#2d313f",
                        'steps': [
                            {'range': [0, 50], 'color': '#d62728'},
                            {'range': [50, 75], 'color': '#ff7f0e'},
                            {'range': [75, 100], 'color': '#2ca02c'}
                        ]
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed", height=280, margin=dict(t=30, b=0, l=10, r=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with res_col2:
                status_color = "green" if predicted_result == "Pass" else "red"
                st.markdown(f"### Predicted Result: <span style='color:{status_color};'>{predicted_result}</span>", unsafe_allow_html=True)
                
                # Educational Recommendations based on prediction
                st.markdown('<div class="rec-box">', unsafe_allow_html=True)
                st.markdown("#### 💡 AI Academic Recommendations:")
                
                recs = []
                if input_attendance < 75.0:
                    recs.append("⚠️ **Improve Attendance**: Attendance is currently below 75%. Prioritizing lecture attendance can heavily impact performance, as attendance is a strong prerequisite for consistent marks.")
                if input_study < 4.0:
                    recs.append("✍️ **Increase Study Time**: Recommended study duration is at least 4-5 hours per day. Try dedicating blocks of focused, undistracted time for self-study.")
                if input_sleep < 6.5:
                    recs.append("💤 **Prioritize Sleep**: Sleep is below 6.5 hours. Studies show cognitive retention peaks at 7-8 hours. Sleep deprivation decreases focus and performance.")
                if input_sleep > 9.0:
                    recs.append("💤 **Optimize Sleep Pattern**: Sleep is over 9 hours. Excess sleep may lead to lethargy. Target an optimal 7.5 hours.")
                
                if not recs:
                    st.write("🎉 **Great academic habits!** The student maintains robust attendance, study time, and sleep. Encourage them to keep up the excellent work.")
                else:
                    for r in recs:
                        st.write(r)
                st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 5: SQL QUERY CONSOLE & DATA EXPORT ---
    with tab5:
        st.subheader("SQL Sandbox / Console")
        st.write("Run read-only SQL queries against the local SQLite database database table `students` directly:")
        
        sql_input = st.text_area(
            "SQL Query Console:", 
            value="SELECT Student_ID, Name, Gender, Department, Final_Exam_Marks, Result FROM students WHERE Result = 'Fail' LIMIT 10;",
            height=120
        )
        
        if st.button("Execute SQL"):
            if sql_input.strip().upper().startswith("SELECT"):
                sql_res = query_db(sql_input)
                if not sql_res.empty:
                    st.success(f"Query returned {len(sql_res)} records:")
                    st.dataframe(sql_res, use_container_width=True)
                else:
                    st.warning("No records returned or query execution failed.")
            else:
                st.error("Security Warning: Only SELECT queries are permitted on this sandbox console.")
                
        st.markdown("---")
        st.subheader("Export Cleaned Data")
        st.write("Download the fully preprocessed and cleaned student performance dataset:")
        
        csv_download = df_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_download,
            file_name="student_performance_cleaned.csv",
            mime="text/csv"
        )
