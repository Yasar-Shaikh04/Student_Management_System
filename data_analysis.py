"""
data_analysis.py
----------------
Data Science Analytics module leveraging Pandas and NumPy.
Performs data inspection, data cleaning, missing-value audit, duplicate detection,
descriptive statistics (mean, median, std, min, max, skewness), correlation analysis,
and multi-dimensional group-by aggregations.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

NUMERIC_COLUMNS = [
    "Age", "Semester", "Attendance", "StudyHoursPerWeek",
    "AssignmentScore", "MidtermScore", "FinalScore",
    "PreviousSemesterScore", "Backlogs", "ExtracurricularHours"
]

CATEGORICAL_COLUMNS = [
    "Gender", "Department", "PerformanceCategory", "RiskStatus"
]

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspects dataset structure, dimensions, data types, and non-null counts."""
    info_list = []
    for col in df.columns:
        info_list.append({
            "Column": col,
            "Non-Null Count": int(df[col].notnull().sum()),
            "Null Count": int(df[col].isnull().sum()),
            "Data Type": str(df[col].dtype),
            "Unique Values": int(df[col].nunique())
        })
    info_df = pd.DataFrame(info_list)
    
    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns_info": info_df,
        "memory_usage_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 2),
        "numeric_columns": [c for c in NUMERIC_COLUMNS if c in df.columns],
        "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    }

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detects missing values across all columns and calculates percentages."""
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": null_counts.values,
        "Missing Percentage (%)": null_percentages.values
    })
    return missing_df[missing_df["Missing Count"] > 0].reset_index(drop=True)

def check_duplicate_records(df: pd.DataFrame, subset: List[str] = None) -> Tuple[int, pd.DataFrame]:
    """Detects duplicate rows based on subset (e.g. StudentID) or entire row."""
    if subset is None:
        subset = ["StudentID"] if "StudentID" in df.columns else None
    
    dups = df[df.duplicated(subset=subset, keep=False)]
    return len(dups), dups

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs data cleaning: removes duplicates, fills missing numeric values with median,
    categorical with mode, and validates boundary constraints."""
    cleaned = df.copy()
    initial_rows = len(cleaned)
    
    # 1. Drop duplicates
    if "StudentID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["StudentID"], keep="first")
    else:
        cleaned = cleaned.drop_duplicates(keep="first")
    dups_removed = initial_rows - len(cleaned)
    
    # 2. Impute missing values
    imputed_count = 0
    for col in cleaned.columns:
        if cleaned[col].isnull().sum() > 0:
            null_count = int(cleaned[col].isnull().sum())
            imputed_count += null_count
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)
            else:
                mode_val = cleaned[col].mode()[0] if not cleaned[col].mode().empty else "Unknown"
                cleaned[col] = cleaned[col].fillna(mode_val)
                
    # 3. Clip numeric bounds to valid ranges
    bounds = {
        "Attendance": (0.0, 100.0),
        "AssignmentScore": (0.0, 100.0),
        "MidtermScore": (0.0, 100.0),
        "FinalScore": (0.0, 100.0),
        "PreviousSemesterScore": (0.0, 100.0),
        "Backlogs": (0, 20),
        "StudyHoursPerWeek": (0.0, 80.0),
        "ExtracurricularHours": (0.0, 40.0)
    }
    clipped_cols = []
    for col, (low, high) in bounds.items():
        if col in cleaned.columns:
            cleaned[col] = np.clip(cleaned[col], low, high)
            clipped_cols.append(col)
            
    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": dups_removed,
        "imputed_missing_values": imputed_count,
        "bounded_columns": clipped_cols
    }
    return cleaned, summary

def calculate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes comprehensive descriptive statistics using Pandas & NumPy:
    Count, Mean, Median, Std Dev, Min, 25%, 50%, 75%, Max, and Skewness."""
    num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    stats_list = []
    
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats_list.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": round(float(np.mean(series)), 2),
            "Median": round(float(np.median(series)), 2),
            "Std Dev": round(float(np.std(series, ddof=1)), 2),
            "Min": round(float(np.min(series)), 2),
            "25% (Q1)": round(float(np.percentile(series, 25)), 2),
            "50% (Q2)": round(float(np.percentile(series, 50)), 2),
            "75% (Q3)": round(float(np.percentile(series, 75)), 2),
            "Max": round(float(np.max(series)), 2),
            "Skewness": round(float(series.skew()), 2)
        })
        
    return pd.DataFrame(stats_list)

def compute_correlation_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Computes Pearson correlation matrix and extracts top correlated pairs."""
    num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    corr_matrix = df[num_cols].corr().round(3)
    
    # Extract non-diagonal top correlations
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            corr_val = float(corr_matrix.loc[col1, col2])
            pairs.append({
                "Feature 1": col1,
                "Feature 2": col2,
                "Correlation": corr_val,
                "Abs Correlation": abs(corr_val),
                "Relationship": "Positive" if corr_val > 0 else "Negative",
                "Strength": "Strong" if abs(corr_val) >= 0.6 else ("Moderate" if abs(corr_val) >= 0.3 else "Weak")
            })
            
    pairs_df = pd.DataFrame(pairs).sort_values(by="Abs Correlation", ascending=False).reset_index(drop=True)
    top_correlations = pairs_df.head(10).to_dict(orient="records")
    return corr_matrix, top_correlations

def perform_group_by_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Executes multi-dimensional group-by aggregations."""
    results = {}
    
    # 1. Department-wise performance
    if "Department" in df.columns:
        dept_grp = df.groupby("Department").agg(
            Student_Count=("StudentID", "count"),
            Avg_Final_Score=("FinalScore", "mean"),
            Median_Final_Score=("FinalScore", "median"),
            Avg_Attendance=("Attendance", "mean"),
            Avg_Study_Hours=("StudyHoursPerWeek", "mean"),
            Avg_Backlogs=("Backlogs", "mean"),
            At_Risk_Students=("RiskStatus", lambda x: (x == "At Risk").sum())
        ).reset_index()
        dept_grp["At_Risk_Rate (%)"] = ((dept_grp["At_Risk_Students"] / dept_grp["Student_Count"]) * 100).round(1)
        dept_grp = dept_grp.round(2).sort_values(by="Avg_Final_Score", ascending=False)
        results["Department_Analysis"] = dept_grp
        
    # 2. Semester-wise performance
    if "Semester" in df.columns:
        sem_grp = df.groupby("Semester").agg(
            Student_Count=("StudentID", "count"),
            Avg_Final_Score=("FinalScore", "mean"),
            Avg_Attendance=("Attendance", "mean"),
            Avg_Study_Hours=("StudyHoursPerWeek", "mean"),
            Total_Backlogs=("Backlogs", "sum")
        ).reset_index().round(2).sort_values(by="Semester")
        results["Semester_Analysis"] = sem_grp

    # 3. Attendance Bracket Analysis
    if "Attendance" in df.columns and "FinalScore" in df.columns:
        bins = [0, 65, 75, 85, 100]
        labels = ["< 65% (Critical)", "65-75% (Warning)", "75-85% (Good)", "> 85% (Excellent)"]
        df_copy = df.copy()
        df_copy["Attendance_Bracket"] = pd.cut(df_copy["Attendance"], bins=bins, labels=labels, right=True)
        att_grp = df_copy.groupby("Attendance_Bracket", observed=False).agg(
            Student_Count=("StudentID", "count"),
            Avg_Final_Score=("FinalScore", "mean"),
            Median_Final_Score=("FinalScore", "median"),
            Avg_Study_Hours=("StudyHoursPerWeek", "mean"),
            At_Risk_Count=("RiskStatus", lambda x: (x == "At Risk").sum())
        ).reset_index().round(2)
        results["Attendance_Bracket_Analysis"] = att_grp

    # 4. Study Hours Bracket Analysis
    if "StudyHoursPerWeek" in df.columns and "FinalScore" in df.columns:
        bins_sh = [0, 10, 18, 28, 100]
        labels_sh = ["< 10 hrs (Low)", "10-18 hrs (Moderate)", "18-28 hrs (High)", "> 28 hrs (Intensive)"]
        df_copy2 = df.copy()
        df_copy2["Study_Bracket"] = pd.cut(df_copy2["StudyHoursPerWeek"], bins=bins_sh, labels=labels_sh, right=True)
        sh_grp = df_copy2.groupby("Study_Bracket", observed=False).agg(
            Student_Count=("StudentID", "count"),
            Avg_Final_Score=("FinalScore", "mean"),
            Avg_Assignment_Score=("AssignmentScore", "mean"),
            Avg_Midterm_Score=("MidtermScore", "mean")
        ).reset_index().round(2)
        results["Study_Bracket_Analysis"] = sh_grp

    return results
