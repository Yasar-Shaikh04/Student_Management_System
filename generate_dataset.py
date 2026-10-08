"""
generate_dataset.py
-------------------
Generates a realistic, synthetic student dataset for the PBL mini-project:
'Agentic Student Management and Data Science Analytics System'.

Produces: data/students.csv (~400 records) with realistic statistical distributions,
logical inter-feature correlations, and derived academic metrics.
"""

import os
import numpy as np
import pandas as pd

def generate_students(n_records: int = 400, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    first_names_male = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna",
        "Ishaan", "Shaurya", "Rohan", "Rahul", "Dev", "Dhruv", "Kabir", "Aryan", "Karthik",
        "Pranav", "Harsh", "Varun", "Aniket", "Gaurav", "Nikhil", "Siddharth", "Yash", "Manish"
    ]
    first_names_female = [
        "Aanya", "Diya", "Saanvi", "Ananya", "Ishita", "Meera", "Pooja", "Neha", "Riya",
        "Kavya", "Tanvi", "Sneha", "Aditi", "Shreya", "Simran", "Priya", "Anushka", "Nandini",
        "Deepika", "Swati", "Shruti", "Tara", "Bhavna", "Kritika", "Preeti", "Radhika", "Divya"
    ]
    last_names = [
        "Sharma", "Verma", "Patel", "Reddy", "Gupta", "Nair", "Iyer", "Rao", "Kumar",
        "Singh", "Chopra", "Deshmukh", "Joshi", "Mehta", "Bose", "Kulkarni", "Mishra",
        "Chatterjee", "Bhattacharya", "Menon", "Pillai", "Das", "Saxena", "Agarwal", "Malhotra"
    ]
    
    departments = [
        "Computer Science",
        "Information Technology",
        "Electronics & Comm",
        "Mechanical Eng",
        "Civil Eng"
    ]
    
    genders = np.random.choice(["Male", "Female"], size=n_records, p=[0.54, 0.46])
    names = []
    for g in genders:
        first = np.random.choice(first_names_male) if g == "Male" else np.random.choice(first_names_female)
        last = np.random.choice(last_names)
        names.append(f"{first} {last}")
        
    student_ids = [f"STU{1001 + i}" for i in range(n_records)]
    
    # Semester: 1 to 8 (uniform or balanced distribution)
    semesters = np.random.choice(range(1, 9), size=n_records)
    
    # Age: typically 18 to 23 based on semester
    ages = [18 + (sem - 1) // 2 + int(np.random.choice([0, 0, 1])) for sem in semesters]
    
    # Department assignment
    dept_choices = np.random.choice(departments, size=n_records, p=[0.30, 0.25, 0.20, 0.13, 0.12])
    
    # Study Hours: normally distributed around 16 hrs/week, clipped to [4, 38]
    study_hours = np.clip(np.random.normal(loc=16.5, scale=5.8, size=n_records), 4.0, 38.0).round(1)
    
    # Attendance: beta-like distribution shifted towards 70-95%, clipped to [40, 99.5]
    # Realistic blend: 75% regular attendees (mean ~84%), 25% lower attendees (mean ~62%)
    regular_mask = np.random.rand(n_records) > 0.22
    attendance = np.zeros(n_records)
    attendance[regular_mask] = np.random.normal(loc=83.5, scale=7.0, size=np.sum(regular_mask))
    attendance[~regular_mask] = np.random.normal(loc=61.0, scale=9.5, size=np.sum(~regular_mask))
    attendance = np.clip(attendance, 38.0, 99.0).round(1)
    
    # Assignment Score: correlated with attendance & study hours
    assignment_latent = 0.45 * attendance + 1.2 * study_hours + np.random.normal(loc=20.0, scale=7.5, size=n_records)
    assignment_scores = np.clip(assignment_latent, 32.0, 99.0).round(1)
    
    # Midterm Score: correlated with study hours and assignments
    midterm_latent = 0.35 * assignment_scores + 0.30 * attendance + 0.8 * study_hours + np.random.normal(loc=8.0, scale=8.0, size=n_records)
    midterm_scores = np.clip(midterm_latent, 30.0, 99.0).round(1)
    
    # Previous Semester Score: consistent student baseline performance
    prev_latent = 0.45 * midterm_scores + 0.35 * assignment_scores + np.random.normal(loc=15.0, scale=6.5, size=n_records)
    prev_scores = np.clip(prev_latent, 35.0, 98.5).round(1)
    
    # Backlogs: lower for high study/attendance, higher for low performers
    backlog_prob = np.clip(1.2 - (midterm_scores / 100.0) - (attendance / 180.0), 0.02, 0.85)
    backlogs = np.array([np.random.poisson(lam=p * 2.2) if p > 0.35 else (1 if np.random.rand() < 0.1 else 0) for p in backlog_prob])
    backlogs = np.clip(backlogs, 0, 6)
    
    # Extracurricular Hours: independent normal distribution [1, 15]
    extracurricular = np.clip(np.random.normal(loc=5.5, scale=2.8, size=n_records), 1.0, 15.0).round(1)
    
    # Final Score derivation:
    # 25% Assignment + 35% Midterm + 20% Attendance factor + 15% Study factor - penalty for backlogs + natural noise
    final_latent = (
        0.25 * assignment_scores +
        0.35 * midterm_scores +
        0.20 * (attendance * 0.85) +
        0.15 * (study_hours * 2.4) +
        np.random.normal(loc=0.0, scale=4.2, size=n_records) -
        (backlogs * 2.8)
    )
    final_scores = np.clip(final_latent, 30.0, 99.5).round(1)
    
    # Performance Category derivation:
    # High Performer: FinalScore >= 80 and Backlogs == 0
    # Good Performer: FinalScore >= 65 and Backlogs <= 1
    # Average Performer: FinalScore >= 50
    # Needs Improvement: FinalScore < 50
    perf_categories = []
    risk_statuses = []
    
    for i in range(n_records):
        score = final_scores[i]
        att = attendance[i]
        bl = backlogs[i]
        
        # Performance Category
        if score >= 80.0 and bl == 0:
            category = "High Performer"
        elif score >= 65.0 and bl <= 1:
            category = "Good Performer"
        elif score >= 50.0:
            category = "Average Performer"
        else:
            category = "Needs Improvement"
        perf_categories.append(category)
        
        # Risk Status (Rule-based: Low Attendance < 75 OR Low Score < 50 OR Backlogs >= 2)
        if att < 75.0 or score < 50.0 or bl >= 2:
            risk = "At Risk"
        else:
            risk = "Safe"
        risk_statuses.append(risk)
        
    df = pd.DataFrame({
        "StudentID": student_ids,
        "Name": names,
        "Age": ages,
        "Gender": genders,
        "Department": dept_choices,
        "Semester": semesters,
        "Attendance": attendance,
        "StudyHoursPerWeek": study_hours,
        "AssignmentScore": assignment_scores,
        "MidtermScore": midterm_scores,
        "FinalScore": final_scores,
        "PreviousSemesterScore": prev_scores,
        "Backlogs": backlogs,
        "ExtracurricularHours": extracurricular,
        "PerformanceCategory": perf_categories,
        "RiskStatus": risk_statuses
    })
    
    return df

def save_dataset(output_path: str = "data/students.csv", n_records: int = 400):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_students(n_records=n_records)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} student records at: {output_path}")
    print("\nDataset Summary Preview:")
    print(df.info())
    print("\nPerformance Categories:")
    print(df["PerformanceCategory"].value_counts())
    print("\nRisk Status Breakdown:")
    print(df["RiskStatus"].value_counts())
    return df

if __name__ == "__main__":
    save_dataset()
