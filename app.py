"""
app.py
------
Streamlit Web Application for:
'Agentic Student Management and Data Science Analytics System'
A PBL Mini-Project for Python for Data Science.

Features 8 cohesive sections:
1. Dashboard
2. Student Management (CRUD)
3. Attendance Management
4. Marks & Performance Management
5. Data Science Analytics
6. Machine Learning Prediction
7. Agentic Analysis (Autonomous Workflow - Beyond Syllabus)
8. Reports & PBL Viva Guide
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modular components
import student_manager as sm
import data_analysis as da
import visualization as viz
import ml_model as mlm
import agent as ag

# Page Configuration
st.set_page_config(
    page_title="Agentic Student Management & Data Science",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .agent-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .badge-syllabus {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-beyond {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to reload session state data
def get_dataset():
    if "students_df" not in st.session_state:
        st.session_state.students_df = sm.load_students_data()
    return st.session_state.students_df

def refresh_dataset(new_df):
    st.session_state.students_df = new_df

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/student-registration.png", width=64)
st.sidebar.title("EduAnalytics")
st.sidebar.caption("Agentic Student Management System")
st.sidebar.markdown("---")

navigation_pages = [
    "📊 Dashboard",
    "👥 Student Management",
    "📅 Attendance Management",
    "📈 Marks & Performance",
    "🔬 Data Science Analytics",
    "🤖 ML Prediction",
    "🧠 Agentic Analysis (Beyond Syllabus)",
    "📑 Reports & PBL Guide"
]

selected_page = st.sidebar.radio("Navigation Menu", navigation_pages, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details:**")
st.sidebar.markdown("• **Course**: Python for Data Science")
st.sidebar.markdown("• **Type**: PBL Mini-Project")
st.sidebar.markdown("• **Core**: Pandas, NumPy, Scikit-learn")
st.sidebar.markdown("• **Novelty**: Autonomous Agentic DS")

df = get_dataset()

# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if selected_page == "📊 Dashboard":
    st.markdown('<div class="main-header">Academic Performance & Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive overview of institutional student records, engagement, and academic risk metrics.</div>', unsafe_allow_html=True)
    
    kpis = sm.get_dashboard_kpis(df)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2563EB;">
            <div class="kpi-title">Total Students</div>
            <div class="kpi-value">{kpis['total_students']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #059669;">
            <div class="kpi-title">Avg Score</div>
            <div class="kpi-value">{kpis['avg_marks']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #0284C7;">
            <div class="kpi-title">Avg Attendance</div>
            <div class="kpi-value">{kpis['avg_attendance']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #D97706;">
            <div class="kpi-title">Avg Study Hrs</div>
            <div class="kpi-value">{kpis['avg_study_hours']}h</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #DC2626;">
            <div class="kpi-title">At Risk</div>
            <div class="kpi-value" style="color: #DC2626;">{kpis['at_risk_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #16A34A;">
            <div class="kpi-title">High Performers</div>
            <div class="kpi-value" style="color: #16A34A;">{kpis['high_performers_count']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dashboard Visualizations
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.pyplot(viz.plot_department_performance(df))
    with row1_c2:
        st.pyplot(viz.plot_performance_distribution(df))
        
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.pyplot(viz.plot_attendance_distribution(df))
    with row2_c2:
        st.pyplot(viz.plot_risk_distribution(df))

# ==============================================================================
# 2. STUDENT MANAGEMENT (CRUD)
# ==============================================================================
elif selected_page == "👥 Student Management":
    st.markdown('<div class="main-header">Student Records Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Create, inspect, filter, update, and manage persistent student records.</div>', unsafe_allow_html=True)
    
    tab_view, tab_add, tab_update, tab_delete = st.tabs([
        "📋 View & Filter Students",
        "➕ Add New Student",
        "✏️ Update Student",
        "🗑️ Delete Student"
    ])
    
    with tab_view:
        st.subheader("Filter & Search Student Records")
        c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1.5])
        with c1:
            search_query = st.text_input("🔍 Search by Student ID or Name", placeholder="e.g. STU1025 or Sharma")
        with c2:
            dept_filter = st.selectbox("Department", ["All"] + sm.DEPARTMENTS)
        with c3:
            cat_filter = st.selectbox("Performance Category", ["All"] + sm.PERFORMANCE_CATEGORIES)
        with c4:
            risk_filter = st.selectbox("Risk Status", ["All", "Safe", "At Risk"])
            
        filtered_df = sm.search_students(df, search_query, dept_filter, cat_filter, risk_filter)
        st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** student records.")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download filtered CSV
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Records (CSV)",
            data=csv_data,
            file_name="filtered_students.csv",
            mime="text/csv"
        )
        
    with tab_add:
        st.subheader("Register a New Student")
        with st.form("add_student_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                new_id = st.text_input("Student ID*", value=f"STU{len(df) + 1001}")
                new_name = st.text_input("Full Name*")
                new_age = st.number_input("Age*", min_value=16, max_value=45, value=20)
                new_gender = st.selectbox("Gender*", sm.GENDERS)
                new_dept = st.selectbox("Department*", sm.DEPARTMENTS)
                new_sem = st.number_input("Semester*", min_value=1, max_value=8, value=4)
                new_att = st.slider("Attendance Percentage (%)*", 0.0, 100.0, 80.0, 0.5)
            with col_b:
                new_study = st.number_input("Study Hours Per Week*", min_value=0.0, max_value=80.0, value=16.0, step=0.5)
                new_assign = st.number_input("Assignment Score (0-100)*", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
                new_midterm = st.number_input("Midterm Score (0-100)*", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
                new_prev = st.number_input("Previous Semester Score (0-100)*", min_value=0.0, max_value=100.0, value=74.0, step=0.5)
                new_backlogs = st.number_input("Active Backlogs*", min_value=0, max_value=20, value=0)
                new_extra = st.number_input("Extracurricular Hours/Week", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
                
            submitted = st.form_submit_button("➕ Submit & Register Student")
            if submitted:
                student_payload = {
                    "StudentID": new_id,
                    "Name": new_name,
                    "Age": new_age,
                    "Gender": new_gender,
                    "Department": new_dept,
                    "Semester": new_sem,
                    "Attendance": new_att,
                    "StudyHoursPerWeek": new_study,
                    "AssignmentScore": new_assign,
                    "MidtermScore": new_midterm,
                    "PreviousSemesterScore": new_prev,
                    "Backlogs": new_backlogs,
                    "ExtracurricularHours": new_extra
                }
                success, msg, updated_df = sm.add_student(df, student_payload)
                if success:
                    refresh_dataset(updated_df)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(f"Failed to add student: {msg}")

    with tab_update:
        st.subheader("Update Existing Student Information")
        student_id_list = df["StudentID"].tolist()
        selected_stu_id = st.selectbox("Select Student to Update", student_id_list)
        
        target_row = df[df["StudentID"] == selected_stu_id].iloc[0]
        
        with st.form("update_student_form"):
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                st.text_input("Student ID (Read-only)", value=target_row["StudentID"], disabled=True)
                u_name = st.text_input("Name", value=target_row["Name"])
                u_age = st.number_input("Age", min_value=16, max_value=45, value=int(target_row["Age"]))
                u_gender = st.selectbox("Gender", sm.GENDERS, index=sm.GENDERS.index(target_row["Gender"]) if target_row["Gender"] in sm.GENDERS else 0)
                u_dept = st.selectbox("Department", sm.DEPARTMENTS, index=sm.DEPARTMENTS.index(target_row["Department"]) if target_row["Department"] in sm.DEPARTMENTS else 0)
                u_sem = st.number_input("Semester", min_value=1, max_value=8, value=int(target_row["Semester"]))
                u_att = st.slider("Attendance (%)", 0.0, 100.0, float(target_row["Attendance"]), 0.5)
            with u_col2:
                u_study = st.number_input("Study Hours/Week", min_value=0.0, max_value=80.0, value=float(target_row["StudyHoursPerWeek"]), step=0.5)
                u_assign = st.number_input("Assignment Score", min_value=0.0, max_value=100.0, value=float(target_row["AssignmentScore"]), step=0.5)
                u_mid = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=float(target_row["MidtermScore"]), step=0.5)
                u_prev = st.number_input("Previous Sem Score", min_value=0.0, max_value=100.0, value=float(target_row["PreviousSemesterScore"]), step=0.5)
                u_bl = st.number_input("Backlogs", min_value=0, max_value=20, value=int(target_row["Backlogs"]))
                u_extra = st.number_input("Extracurricular Hours", min_value=0.0, max_value=30.0, value=float(target_row["ExtracurricularHours"]), step=0.5)

            u_submit = st.form_submit_button("💾 Save Updates")
            if u_submit:
                update_payload = {
                    "StudentID": selected_stu_id,
                    "Name": u_name,
                    "Age": u_age,
                    "Gender": u_gender,
                    "Department": u_dept,
                    "Semester": u_sem,
                    "Attendance": u_att,
                    "StudyHoursPerWeek": u_study,
                    "AssignmentScore": u_assign,
                    "MidtermScore": u_mid,
                    "PreviousSemesterScore": u_prev,
                    "Backlogs": u_bl,
                    "ExtracurricularHours": u_extra
                }
                success, msg, updated_df = sm.update_student(df, selected_stu_id, update_payload)
                if success:
                    refresh_dataset(updated_df)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with tab_delete:
        st.subheader("Delete Student Record")
        del_student_id = st.selectbox("Select Student ID to Remove", df["StudentID"].tolist(), key="del_select")
        del_row = df[df["StudentID"] == del_student_id].iloc[0]
        
        st.warning(f"Are you sure you want to delete **{del_row['Name']}** ({del_student_id}) from **{del_row['Department']}**?")
        st.info(f"Student Stats: Attendance: {del_row['Attendance']}% | Final Score: {del_row['FinalScore']} | Backlogs: {del_row['Backlogs']}")
        
        col_d1, col_d2 = st.columns([1, 4])
        with col_d1:
            if st.button("🚨 Confirm Delete", type="primary"):
                success, msg, updated_df = sm.delete_student(df, del_student_id)
                if success:
                    refresh_dataset(updated_df)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

# ==============================================================================
# 3. ATTENDANCE MANAGEMENT
# ==============================================================================
elif selected_page == "📅 Attendance Management":
    st.markdown('<div class="main-header">Attendance Management & Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze attendance distributions, configure warning thresholds, and identify students at risk.</div>', unsafe_allow_html=True)
    
    threshold = st.slider("Regulatory Attendance Warning Threshold (%)", min_value=50.0, max_value=90.0, value=75.0, step=1.0)
    
    low_att_df = df[df["Attendance"] < threshold]
    low_att_pct = (len(low_att_df) / len(df)) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Institutional Avg Attendance", f"{df['Attendance'].mean():.1f}%")
    with c2:
        st.metric("Median Attendance", f"{df['Attendance'].median():.1f}%")
    with c3:
        st.metric(f"Students Below {threshold}%", f"{len(low_att_df)}", delta=f"{low_att_pct:.1f}%", delta_color="inverse")
    with c4:
        st.metric("Attendance Std Dev", f"{df['Attendance'].std():.2f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_plot, col_alert = st.columns([1.2, 1])
    with col_plot:
        st.pyplot(viz.plot_attendance_distribution(df, threshold=threshold))
    with col_alert:
        st.subheader(f"⚠️ Students with Attendance < {threshold}%")
        st.dataframe(
            low_att_df[["StudentID", "Name", "Department", "Attendance", "FinalScore", "RiskStatus"]].sort_values(by="Attendance"),
            use_container_width=True,
            hide_index=True
        )

# ==============================================================================
# 4. MARKS & PERFORMANCE MANAGEMENT
# ==============================================================================
elif selected_page == "📈 Marks & Performance":
    st.markdown('<div class="main-header">Marks & Academic Performance Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of continuous assessments, midterm performance, and final score derivations.</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Institutional Average", f"{df['FinalScore'].mean():.1f} / 100")
    with m2:
        st.metric("Highest Final Score", f"{df['FinalScore'].max():.1f}")
    with m3:
        st.metric("Lowest Final Score", f"{df['FinalScore'].min():.1f}")
    with m4:
        st.metric("Pass Rate (Score ≥ 50)", f"{(df['FinalScore'] >= 50).mean()*100:.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    r1, r2 = st.columns(2)
    with r1:
        st.pyplot(viz.plot_marks_distribution(df))
    with r2:
        st.pyplot(viz.plot_study_hours_vs_performance(df))
        
    st.subheader("Formula & Score Derivation Logic")
    st.info("""
    **Objective Scoring Model Formula**:
    `FinalScore = 0.25 * Assignment + 0.35 * Midterm + 0.20 * (Attendance * 0.85) + 0.15 * (StudyHours * 2.4) - (Backlogs * 2.5)`
    - High Performer: Score ≥ 80 and 0 Backlogs
    - Good Performer: Score ≥ 65 and ≤ 1 Backlog
    - Average Performer: Score ≥ 50
    - Needs Improvement: Score < 50
    """)

# ==============================================================================
# 5. DATA SCIENCE ANALYTICS
# ==============================================================================
elif selected_page == "🔬 Data Science Analytics":
    st.markdown('<div class="main-header">Data Science Analytics & Statistical Profiling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrating core syllabus competencies using Pandas, NumPy, Matplotlib, and Seaborn.</div>', unsafe_allow_html=True)
    
    tab_inspect, tab_stats, tab_corr, tab_grp = st.tabs([
        "🔍 Data Inspection & Hygiene",
        "📊 Descriptive Statistics",
        "🔗 Correlation Analysis",
        "📑 Group-By Aggregations"
    ])
    
    with tab_inspect:
        st.subheader("Dataset Architecture & Schema")
        inspect_res = da.inspect_dataset(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", inspect_res["total_rows"])
        c2.metric("Total Features", inspect_res["total_columns"])
        c3.metric("Memory Usage", f"{inspect_res['memory_usage_kb']} KB")
        
        st.markdown("**Column Schema & Non-Null Audit:**")
        st.dataframe(inspect_res["columns_info"], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Data Cleaning & Anomaly Checks")
        missing_df = da.check_missing_values(df)
        dup_count, _ = da.check_duplicate_records(df)
        
        col_m, col_d = st.columns(2)
        with col_m:
            if len(missing_df) == 0:
                st.success("✅ **Zero Missing Values**: All column records are 100% complete.")
            else:
                st.warning(f"⚠️ Found {len(missing_df)} columns with missing entries.")
                st.dataframe(missing_df, hide_index=True)
        with col_d:
            if dup_count == 0:
                st.success("✅ **Zero Duplicate Records**: StudentID uniqueness constraint satisfied.")
            else:
                st.warning(f"⚠️ Found {dup_count} duplicate rows.")

    with tab_stats:
        st.subheader("Descriptive Statistical Profiling (Pandas / NumPy)")
        stats_df = da.calculate_descriptive_statistics(df)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        csv_stats = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Statistics Summary (CSV)", csv_stats, "descriptive_statistics.csv", "text/csv")

    with tab_corr:
        st.subheader("Correlation Heatmap & Key Relationships")
        c_fig, c_pairs = da.compute_correlation_matrix(df)
        
        col_h, col_t = st.columns([1.3, 1])
        with col_h:
            st.pyplot(viz.plot_correlation_heatmap(df))
        with col_t:
            st.markdown("**Top Correlated Feature Pairs:**")
            pairs_df = pd.DataFrame(c_pairs)[["Feature 1", "Feature 2", "Correlation", "Strength"]]
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Visual Relationship: Attendance vs Final Score:**")
            st.pyplot(viz.plot_attendance_vs_marks(df))

    with tab_grp:
        st.subheader("Multi-Dimensional Group-By Aggregations")
        grp_results = da.perform_group_by_analysis(df)
        
        if "Department_Analysis" in grp_results:
            st.markdown("#### 1. Department-Wise Academic Metrics")
            st.dataframe(grp_results["Department_Analysis"], use_container_width=True, hide_index=True)
            
        if "Attendance_Bracket_Analysis" in grp_results:
            st.markdown("#### 2. Attendance Cohort Analysis")
            st.dataframe(grp_results["Attendance_Bracket_Analysis"], use_container_width=True, hide_index=True)
            
        if "Study_Bracket_Analysis" in grp_results:
            st.markdown("#### 3. Weekly Study Hours Tier Analysis")
            st.dataframe(grp_results["Study_Bracket_Analysis"], use_container_width=True, hide_index=True)

# ==============================================================================
# 6. MACHINE LEARNING PREDICTION
# ==============================================================================
elif selected_page == "🤖 ML Prediction":
    st.markdown('<div class="main-header">Supervised Machine Learning: Student Performance Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting student Performance Category using explainable Scikit-learn classification models.</div>', unsafe_allow_html=True)
    
    col_ctrl, col_metrics = st.columns([1, 2])
    with col_ctrl:
        st.subheader("Model Configuration")
        model_choice = st.selectbox("Classifier Algorithm", ["Decision Tree", "Random Forest"])
        tree_depth = st.slider("Maximum Tree Depth", 2, 8, 4)
        
        if "predictor" not in st.session_state or st.session_state.get("last_model_choice") != (model_choice, tree_depth):
            with st.spinner("Training model..."):
                predictor = mlm.StudentPerformancePredictor(model_type=model_choice, max_depth=tree_depth)
                predictor.train(df)
                st.session_state.predictor = predictor
                st.session_state.last_model_choice = (model_choice, tree_depth)
        predictor = st.session_state.predictor

    with col_metrics:
        st.subheader("Model Evaluation Metrics")
        acc1, acc2, acc3 = st.columns(3)
        acc1.metric("Training Accuracy", f"{predictor.train_accuracy}%")
        acc2.metric("Testing Accuracy (Holdout)", f"{predictor.test_accuracy}%")
        acc3.metric("Split Ratio", "75% Train / 25% Test")
        
    st.markdown("---")
    
    col_fi, col_cm = st.columns(2)
    with col_fi:
        st.pyplot(predictor.plot_feature_importance())
    with col_cm:
        st.pyplot(predictor.plot_confusion_matrix())
        
    st.markdown("---")
    st.subheader("🔮 Interactive 'What-If' Performance Predictor")
    st.markdown("Adjust student attributes to simulate how engagement factors alter predicted academic outcomes:")
    
    with st.form("what_if_prediction_form"):
        pred_c1, pred_c2, pred_c3 = st.columns(3)
        with pred_c1:
            inp_att = st.slider("Attendance (%)", 0.0, 100.0, 78.0, 1.0)
            inp_study = st.slider("Weekly Study Hours", 0.0, 50.0, 16.0, 0.5)
        with pred_c2:
            inp_assign = st.slider("Assignment Score (0-100)", 0.0, 100.0, 75.0, 1.0)
            inp_midterm = st.slider("Midterm Score (0-100)", 0.0, 100.0, 70.0, 1.0)
        with pred_c3:
            inp_prev = st.slider("Previous Semester Score", 0.0, 100.0, 72.0, 1.0)
            inp_backlogs = st.number_input("Active Backlogs", min_value=0, max_value=10, value=0)
            
        predict_btn = st.form_submit_button("🎯 Predict Performance Category")
        
        if predict_btn:
            input_dict = {
                "Attendance": inp_att,
                "StudyHoursPerWeek": inp_study,
                "AssignmentScore": inp_assign,
                "MidtermScore": inp_midterm,
                "PreviousSemesterScore": inp_prev,
                "Backlogs": inp_backlogs
            }
            res = predictor.predict_single(input_dict)
            
            st.success(f"### Predicted Performance Category: **{res['prediction']}** (Confidence: {res['confidence']}%)")
            
            st.markdown("**Class Probability Breakdown:**")
            p_cols = st.columns(len(res["probabilities"]))
            for pcol, (cls_name, prob_val) in zip(p_cols, res["probabilities"].items()):
                pcol.metric(cls_name, f"{prob_val}%")

    st.caption("ℹ️ *Disclaimer: This machine learning model is intended strictly for PBL academic mini-project demonstration.*")

# ==============================================================================
# 7. AGENTIC DATA SCIENCE (BEYOND SYLLABUS)
# ==============================================================================
elif selected_page == "🧠 Agentic Analysis (Beyond Syllabus)":
    st.markdown('<div class="main-header">Autonomous Agentic Data Science Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
        <span class="badge-beyond">BEYOND SYLLABUS NOVELTY</span>
        An autonomous multi-stage workflow controller that inspects data, makes cleaning & analytical decisions, triggers targeted EDA & ML, and synthesizes dynamic evidence-based insights.
    </div>
    """, unsafe_allow_html=True)
    
    # Workflow Architecture Diagram
    st.markdown("""
    ```text
    ┌───────────────────────────┐
    │  Raw Student Dataset CSV  │
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │  Data Understanding Agent │ ➔ Infers feature cardinality, types, memory profile
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │ Data Quality & Audit Agent│ ➔ Evaluates missingness, duplicates, out-of-range bounds
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │ Strategic Decision Engine │ ➔ Autonomously decides cleaning needs, ML viability, top correlations
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │ Autonomous EDA & ML Exec  │ ➔ Runs targeted descriptive stats, group-bys, and model training
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │ Dynamic Insight Synthesis │ ➔ Computes dynamic statistical facts & actionable recommendations
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │ Executive Summary & Trace │ ➔ Renders transparent execution DAG and decision trace
    └───────────────────────────┘
    ```
    """)
    
    col_run, col_status = st.columns([1, 2])
    with col_run:
        run_agent_btn = st.button("🚀 Run Autonomous Agentic Pipeline", type="primary", use_container_width=True)
    
    if run_agent_btn or "agent_results" in st.session_state:
        if run_agent_btn:
            with st.spinner("🤖 Autonomous Agent analyzing dataset and executing decision pipeline..."):
                controller = ag.AgenticWorkflowController()
                agent_results = controller.run_pipeline(df)
                st.session_state.agent_results = agent_results
                st.session_state.agent_controller = controller
                
        agent_results = st.session_state.agent_results
        controller = st.session_state.agent_controller
        
        st.success("✅ **Autonomous Workflow Completed Successfully!**")
        
        # Section A: Step-by-Step Decision Trace
        st.subheader("1. Autonomous Agent Execution Trace (DAG)")
        trace_data = []
        for step in agent_results["trace"]:
            trace_data.append({
                "Step": step.step_id,
                "Stage": step.stage,
                "Status": step.status,
                "Agent Thought": step.agent_thought,
                "Action Taken": step.action_taken,
                "Decision / Branch": step.decision_taken
            })
        st.dataframe(pd.DataFrame(trace_data), use_container_width=True, hide_index=True)
        
        # Section B: Step Details Expanders
        st.markdown("#### 🔬 Inspect Step-by-Step Agent Decisions")
        for step in agent_results["trace"]:
            with st.expander(f"Stage {step.step_id}: {step.stage} [{step.status}]"):
                st.markdown(f"**🧠 Agent Thought:** *{step.agent_thought}*")
                st.markdown(f"**⚙️ Action Executed:** {step.action_taken}")
                if step.decision_taken:
                    st.markdown(f"**🔀 Autonomous Decision:** `{step.decision_taken}`")
                if step.findings:
                    st.json(step.findings)

        # Section C: Dynamic Insights
        st.markdown("---")
        st.subheader("2. Dynamically Synthesized Academic Insights")
        st.caption("Insights are mathematically derived on the fly by the agent from the active dataset:")
        
        for ins in agent_results["insights"]:
            st.markdown(f"""
            <div class="agent-box">
                <h4>{ins['icon']} {ins['title']}</h4>
                <p><strong>Category:</strong> {ins['category']}</p>
                <p>{ins['detail']}</p>
                <div style="background-color: #DBEAFE; padding: 8px 12px; border-radius: 4px; color: #1E40AF; font-size: 0.9rem;">
                    💡 <strong>Autonomous Actionable Recommendation:</strong> {ins['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Section D: Export Agent Summary
        st.markdown("---")
        summary_md = controller.generate_markdown_summary()
        st.download_button(
            label="📥 Download Agentic Executive Summary (Markdown)",
            data=summary_md,
            file_name="Agentic_Data_Science_Summary.md",
            mime="text/markdown"
        )

# ==============================================================================
# 8. REPORTS & PBL VIVA GUIDE
# ==============================================================================
elif selected_page == "📑 Reports & PBL Guide":
    st.markdown('<div class="main-header">Academic Report & PBL Evaluation Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive documentation, syllabus alignment, viva presentation checklist, and FAQs.</div>', unsafe_allow_html=True)
    
    tab_summary, tab_checklist, tab_viva, tab_full_report = st.tabs([
        "📋 Project Summary",
        "⏱️ 5-Minute Demonstration Checklist",
        "🗣️ Viva Q&A Guide",
        "📄 Full Academic Report (25 Sections)"
    ])
    
    with tab_summary:
        st.subheader("Syllabus vs. Beyond-Syllabus Alignment")
        comp_df = pd.DataFrame([
            {"Domain": "Core Syllabus", "Concept": "Python Basics & Object-Oriented Programming", "Implementation": "Classes, functions, type validation in student_manager.py"},
            {"Domain": "Core Syllabus", "Concept": "Pandas Data Manipulation", "Implementation": "DataFrame CRUD, multi-criteria filtering, group-by aggregations"},
            {"Domain": "Core Syllabus", "Concept": "NumPy Numerical Operations", "Implementation": "Vectorized score derivation, array clipping, percentile statistics"},
            {"Domain": "Core Syllabus", "Concept": "Data Cleaning & Hygiene", "Implementation": "Deduplication, median/mode imputation, constraint bounding"},
            {"Domain": "Core Syllabus", "Concept": "Data Visualization", "Implementation": "Matplotlib & Seaborn plots (KDE, bar, regplot, heatmaps)"},
            {"Domain": "Core Syllabus", "Concept": "Supervised Machine Learning", "Implementation": "Scikit-learn Decision Tree & Random Forest performance classifiers"},
            {"Domain": "Beyond Syllabus (Novelty)", "Concept": "Agentic Data Science & Autonomous Workflows", "Implementation": "Autonomous multi-stage workflow controller (agent.py) with dynamic insight generation"}
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with tab_checklist:
        st.subheader("5–10 Minute Viva Evaluation Demonstration Sequence")
        st.markdown("""
        Follow this exact sequence for a high-scoring demonstration during project evaluation:
        
        1. **Introduction (1 min)**:
           - Open the **Dashboard**. Point out institutional KPIs: total students (~400), average attendance, and at-risk student count.
        2. **Student Management & CRUD (1.5 min)**:
           - Navigate to **Student Management**. Demonstrate searching a student by name/ID.
           - Go to **Add Student** tab: add a test student with low attendance (<60%) and show that they are immediately flagged as *At Risk*.
        3. **Attendance & Risk Analysis (1 min)**:
           - Open **Attendance Management**. Adjust the regulatory threshold slider to 80% and show how the low-attendance table updates dynamically.
        4. **Data Science & Statistics (1.5 min)**:
           - Navigate to **Data Science Analytics**. Show the descriptive statistics table (mean, median, skewness), correlation heatmap, and department group-by table.
        5. **Machine Learning Model (1.5 min)**:
           - Navigate to **ML Prediction**. Show train/test accuracy and the feature importance chart.
           - Move the sliders in the **What-If Predictor** to show how attendance and study hours change the predicted performance category.
        6. **Beyond-Syllabus: Agentic Data Science (2.5 min)**:
           - Navigate to **Agentic Analysis**. Click **'Run Autonomous Agentic Pipeline'**.
           - Explain the autonomous decision DAG: how the agent audited data quality, determined ML viability, executed targeted EDA, and derived dynamic, evidence-backed insights.
        """)

    with tab_viva:
        st.subheader("Top Viva Questions & Recommended Answers")
        
        viva_qa = [
            ("Q1: Why did you use a Decision Tree for the ML component?",
             "Decision Trees provide complete white-box explainability. By constraining max_depth=4, we can explain the exact decision paths (e.g. if Attendance >= 75% and Backlogs == 0) to faculty evaluators, avoiding opaque black-box models."),
            
            ("Q2: How does the Agentic Data Science module differ from a regular script?",
             "A conventional script executes a hardcoded static pipeline regardless of data traits. Our Agentic Workflow Controller dynamically audits data quality, autonomously decides whether cleaning is required, selects top correlations for deeper inspection, verifies ML prerequisites, and generates evidence-based insights dynamically calculated on the active dataset."),
             
            ("Q3: How is the synthetic dataset generated and is it realistic?",
             "The dataset (data/students.csv) is generated using NumPy with realistic correlations: attendance and study hours positively correlate with assessment scores, while backlogs penalize final scores. Performance categories and risk statuses are derived objectively rather than assigned randomly."),
             
            ("Q4: What are the criteria for flagging a student as 'At Risk'?",
             "A rule-based threshold approach: Attendance < 75%, Final Score < 50, or Backlogs >= 2. This represents standard academic regulation across engineering colleges."),
             
            ("Q5: Why did you avoid heavy databases and complex frameworks?",
             "In accordance with PBL design principles, we used CSV files and Pandas for high portability and rapid demonstration without external database dependencies or complex setup overhead.")
        ]
        
        for q, a in viva_qa:
            with st.expander(q):
                st.write(a)

    with tab_full_report:
        st.subheader("Complete 25-Section PBL Project Report")
        st.markdown("The complete academic project report is documented in `docs/PBL_PROJECT_REPORT.md`.")
        
        import os
        report_path = os.path.join(os.path.dirname(__file__), "docs", "PBL_PROJECT_REPORT.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
            st.download_button(
                label="📥 Download Full PBL Report (.md)",
                data=report_text,
                file_name="Agentic_Student_Management_PBL_Report.md",
                mime="text/markdown"
            )
            st.markdown(report_text[:2500] + "\n\n*(Full report available in download above and in docs/ folder)*")
        else:
            st.info("Report file will be generated in docs/PBL_PROJECT_REPORT.md.")
