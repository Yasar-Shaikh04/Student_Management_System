"""
student_manager.py
------------------
Handles all CRUD (Create, Read, Update, Delete) operations, CSV data persistence,
input validation, filtering, and rule-based Student Risk Analysis.
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "students.csv")

DEPARTMENTS = [
    "Computer Science",
    "Information Technology",
    "Electronics & Comm",
    "Mechanical Eng",
    "Civil Eng"
]

GENDERS = ["Male", "Female", "Other"]

PERFORMANCE_CATEGORIES = [
    "High Performer",
    "Good Performer",
    "Average Performer",
    "Needs Improvement"
]

def derive_final_score(assignment: float, midterm: float, attendance: float, study_hours: float, backlogs: int) -> float:
    """Derives a balanced final score from assessments, attendance, and study factor."""
    raw_score = (
        0.25 * float(assignment) +
        0.35 * float(midterm) +
        0.20 * (float(attendance) * 0.85) +
        0.15 * (float(study_hours) * 2.4) -
        (int(backlogs) * 2.5)
    )
    return round(float(np.clip(raw_score, 0.0, 100.0)), 1)

def derive_performance_category(final_score: float, backlogs: int) -> str:
    """Classifies student performance objectively based on final score and backlogs."""
    if final_score >= 80.0 and backlogs == 0:
        return "High Performer"
    elif final_score >= 65.0 and backlogs <= 1:
        return "Good Performer"
    elif final_score >= 50.0:
        return "Average Performer"
    else:
        return "Needs Improvement"

def derive_risk_status(
    attendance: float,
    final_score: float,
    backlogs: int,
    att_thresh: float = 75.0,
    score_thresh: float = 50.0,
    backlog_thresh: int = 2
) -> str:
    """Flags a student as 'At Risk' if attendance, marks, or backlogs cross educational warning thresholds."""
    if float(attendance) < att_thresh or float(final_score) < score_thresh or int(backlogs) >= backlog_thresh:
        return "At Risk"
    return "Safe"

def load_students_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads student data from CSV. Generates a fresh dataset if file doesn't exist."""
    if not os.path.exists(csv_path):
        from generate_dataset import save_dataset
        return save_dataset(output_path=csv_path)
    
    df = pd.read_csv(csv_path)
    # Ensure proper data types
    numeric_cols = [
        "Age", "Semester", "Attendance", "StudyHoursPerWeek",
        "AssignmentScore", "MidtermScore", "FinalScore",
        "PreviousSemesterScore", "Backlogs", "ExtracurricularHours"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def save_students_data(df: pd.DataFrame, csv_path: str = DATA_PATH) -> bool:
    """Saves DataFrame back to CSV safely."""
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving data to {csv_path}: {e}")
        return False

def validate_student_record(data: Dict[str, Any], existing_ids: list, is_update: bool = False) -> Tuple[bool, str]:
    """Validates inputs for adding or updating a student."""
    # Check ID
    student_id = str(data.get("StudentID", "")).strip().upper()
    if not student_id:
        return False, "Student ID cannot be empty."
    if not is_update and student_id in existing_ids:
        return False, f"Student ID '{student_id}' already exists. Please use a unique ID."
    
    # Check Name
    name = str(data.get("Name", "")).strip()
    if len(name) < 2:
        return False, "Name must be at least 2 characters long."
    
    # Check Age
    try:
        age = int(data.get("Age", 0))
        if age < 16 or age > 45:
            return False, "Age must be between 16 and 45."
    except (ValueError, TypeError):
        return False, "Age must be a valid integer."
        
    # Check Semester
    try:
        sem = int(data.get("Semester", 0))
        if sem < 1 or sem > 8:
            return False, "Semester must be between 1 and 8."
    except (ValueError, TypeError):
        return False, "Semester must be an integer between 1 and 8."

    # Check Attendance
    try:
        att = float(data.get("Attendance", 0))
        if att < 0.0 or att > 100.0:
            return False, "Attendance must be between 0.0% and 100.0%."
    except (ValueError, TypeError):
        return False, "Attendance must be a valid number."

    # Check Study Hours
    try:
        sh = float(data.get("StudyHoursPerWeek", 0))
        if sh < 0.0 or sh > 80.0:
            return False, "Study Hours per week must be between 0 and 80."
    except (ValueError, TypeError):
        return False, "Study hours must be a valid number."

    # Check Assessment Scores
    for score_col in ["AssignmentScore", "MidtermScore", "PreviousSemesterScore"]:
        try:
            val = float(data.get(score_col, 0))
            if val < 0.0 or val > 100.0:
                return False, f"{score_col} must be between 0.0 and 100.0."
        except (ValueError, TypeError):
            return False, f"{score_col} must be a valid numeric score."

    # Check Backlogs
    try:
        bl = int(data.get("Backlogs", 0))
        if bl < 0 or bl > 20:
            return False, "Backlogs must be a non-negative integer (0-20)."
    except (ValueError, TypeError):
        return False, "Backlogs must be a valid integer."

    return True, "Validation successful."

def add_student(df: pd.DataFrame, student_data: Dict[str, Any], csv_path: str = DATA_PATH) -> Tuple[bool, str, pd.DataFrame]:
    """Adds a new student record to the DataFrame and persists to CSV."""
    existing_ids = df["StudentID"].astype(str).tolist() if "StudentID" in df.columns else []
    is_valid, msg = validate_student_record(student_data, existing_ids, is_update=False)
    if not is_valid:
        return False, msg, df

    # Prepare values
    student_id = str(student_data["StudentID"]).strip().upper()
    name = str(student_data["Name"]).strip()
    age = int(student_data["Age"])
    gender = str(student_data.get("Gender", "Male"))
    dept = str(student_data.get("Department", "Computer Science"))
    sem = int(student_data["Semester"])
    att = float(student_data["Attendance"])
    study_hrs = float(student_data["StudyHoursPerWeek"])
    assignment = float(student_data["AssignmentScore"])
    midterm = float(student_data["MidtermScore"])
    prev_score = float(student_data.get("PreviousSemesterScore", midterm))
    backlogs = int(student_data.get("Backlogs", 0))
    extra = float(student_data.get("ExtracurricularHours", 3.0))

    final_score = student_data.get("FinalScore")
    if final_score is None or float(final_score) <= 0:
        final_score = derive_final_score(assignment, midterm, att, study_hrs, backlogs)
    else:
        final_score = round(float(final_score), 1)

    category = derive_performance_category(final_score, backlogs)
    risk = derive_risk_status(att, final_score, backlogs)

    new_row = {
        "StudentID": student_id,
        "Name": name,
        "Age": age,
        "Gender": gender,
        "Department": dept,
        "Semester": sem,
        "Attendance": att,
        "StudyHoursPerWeek": study_hrs,
        "AssignmentScore": assignment,
        "MidtermScore": midterm,
        "FinalScore": final_score,
        "PreviousSemesterScore": prev_score,
        "Backlogs": backlogs,
        "ExtracurricularHours": extra,
        "PerformanceCategory": category,
        "RiskStatus": risk
    }

    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_students_data(updated_df, csv_path)
    return True, f"Student {name} ({student_id}) successfully added.", updated_df

def update_student(df: pd.DataFrame, student_id: str, updated_data: Dict[str, Any], csv_path: str = DATA_PATH) -> Tuple[bool, str, pd.DataFrame]:
    """Updates an existing student record."""
    student_id = str(student_id).strip().upper()
    idx_matches = df.index[df["StudentID"] == student_id].tolist()
    if not idx_matches:
        return False, f"Student with ID '{student_id}' not found.", df

    idx = idx_matches[0]
    existing_ids = df["StudentID"].astype(str).tolist()
    
    # Merge existing student record with updated fields
    merged_data = df.loc[idx].to_dict()
    merged_data.update(updated_data)

    is_valid, msg = validate_student_record(merged_data, existing_ids, is_update=True)
    if not is_valid:
        return False, msg, df

    # Recalculate derived fields
    assignment = float(merged_data.get("AssignmentScore", df.at[idx, "AssignmentScore"]))
    midterm = float(merged_data.get("MidtermScore", df.at[idx, "MidtermScore"]))
    att = float(merged_data.get("Attendance", df.at[idx, "Attendance"]))
    study_hrs = float(merged_data.get("StudyHoursPerWeek", df.at[idx, "StudyHoursPerWeek"]))
    backlogs = int(merged_data.get("Backlogs", df.at[idx, "Backlogs"]))

    # Derive final score if not explicitly fixed
    final_score = updated_data.get("FinalScore")
    if final_score is None:
        final_score = derive_final_score(assignment, midterm, att, study_hrs, backlogs)
    else:
        final_score = round(float(final_score), 1)

    category = derive_performance_category(final_score, backlogs)
    risk = derive_risk_status(att, final_score, backlogs)

    # Apply updates
    for key, val in merged_data.items():
        if key in df.columns:
            df.at[idx, key] = val

    df.at[idx, "FinalScore"] = final_score
    df.at[idx, "PerformanceCategory"] = category
    df.at[idx, "RiskStatus"] = risk

    save_students_data(df, csv_path)
    return True, f"Student record for ID '{student_id}' successfully updated.", df

def delete_student(df: pd.DataFrame, student_id: str, csv_path: str = DATA_PATH) -> Tuple[bool, str, pd.DataFrame]:
    """Deletes a student record by StudentID."""
    student_id = str(student_id).strip().upper()
    idx_matches = df.index[df["StudentID"] == student_id].tolist()
    if not idx_matches:
        return False, f"Student with ID '{student_id}' not found.", df

    deleted_name = df.at[idx_matches[0], "Name"]
    updated_df = df.drop(index=idx_matches).reset_index(drop=True)
    save_students_data(updated_df, csv_path)
    return True, f"Student '{deleted_name}' ({student_id}) has been removed.", updated_df

def search_students(
    df: pd.DataFrame,
    query: str = "",
    department: str = "All",
    category: str = "All",
    risk_status: str = "All",
    semester: int = 0
) -> pd.DataFrame:
    """Performs multi-criteria filtering across student records."""
    filtered = df.copy()
    
    if query.strip():
        q = query.strip().lower()
        mask = (
            filtered["StudentID"].astype(str).str.lower().str.contains(q) |
            filtered["Name"].astype(str).str.lower().str.contains(q)
        )
        filtered = filtered[mask]
        
    if department != "All":
        filtered = filtered[filtered["Department"] == department]
        
    if category != "All":
        filtered = filtered[filtered["PerformanceCategory"] == category]
        
    if risk_status != "All":
        filtered = filtered[filtered["RiskStatus"] == risk_status]
        
    if semester > 0:
        filtered = filtered[filtered["Semester"] == semester]
        
    return filtered

def get_dashboard_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates top-level KPI metrics for dashboard cards."""
    total = len(df)
    if total == 0:
        return {
            "total_students": 0, "avg_marks": 0.0, "avg_attendance": 0.0,
            "avg_study_hours": 0.0, "at_risk_count": 0, "high_performers_count": 0
        }
    return {
        "total_students": total,
        "avg_marks": round(float(df["FinalScore"].mean()), 1),
        "avg_attendance": round(float(df["Attendance"].mean()), 1),
        "avg_study_hours": round(float(df["StudyHoursPerWeek"].mean()), 1),
        "at_risk_count": int((df["RiskStatus"] == "At Risk").sum()),
        "high_performers_count": int((df["PerformanceCategory"] == "High Performer").sum())
    }
