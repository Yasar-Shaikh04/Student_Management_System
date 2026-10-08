# Project-Based Learning (PBL) Mini-Project Report

**Course**: Python for Data Science  
**Project Title**: Agentic Student Management and Data Science Analytics System  
**Academic Year**: 2026–2027  

---

## 1. Title
**Agentic Student Management and Data Science Analytics System**

---

## 2. Abstract
In contemporary higher education, institutional student tracking and performance prediction are vital for early intervention and curriculum optimization. Conventional student management systems act as passive record-keepers, offering rudimentary tabular data without dynamic analytics or decision intelligence. This project introduces the **Agentic Student Management and Data Science Analytics System**, a unified web-based application built using Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. The core syllabus implementation provides complete student record lifecycle management (CRUD), regulatory attendance threshold tracking, assessment score derivation, descriptive statistical profiling, correlation heatmaps, and explainable supervised machine learning via Decision Trees. Beyond the standard curriculum, the project pioneers an **Autonomous Agentic Data Science Workflow Controller**. This agent autonomously assesses data quality, takes cleaning and analytical decisions, executes targeted Exploratory Data Analysis (EDA) and predictive modeling, and synthesizes dynamic, evidence-based insights directly from active dataset distributions. The system uses a realistic synthetic dataset of 400 engineering students, demonstrating that automated agentic workflows enhance analytical efficiency while maintaining full explainability during academic evaluation.

---

## 3. Introduction
Academic institutions generate substantial volumes of student engagement and evaluation metrics, including attendance percentages, assignment submissions, midterm scores, study hours, and backlog histories. Extracting actionable intelligence from these records requires a blend of reliable data management and advanced statistical computing.

Python has emerged as the premier language for data science due to its rich ecosystem of libraries including Pandas, NumPy, Matplotlib, and Scikit-learn. Furthermore, modern data science is transitioning from static, human-triggered scripting toward **Agentic Data Science**, where autonomous controllers evaluate dataset characteristics and decide what analytical operations to execute. This project demonstrates this paradigm shift within an educational domain.

---

## 4. Problem Statement
Traditional academic management systems suffer from several key limitations:
1. **Siloed Records**: Data is stored passively with no immediate statistical profiling.
2. **Delayed Intervention**: Students at risk of academic failure or regulatory detention are identified only after semester examinations conclude.
3. **Manual Analysis Overhead**: Faculty members must manually compute averages, correlation matrices, and risk lists.
4. **Lack of Explainable Prediction**: Black-box predictive models fail to provide interpretable explanations suitable for academic counseling and college evaluations.

---

## 5. Objectives
1. Develop a functional **Student Management System** supporting Create, Read, Update, and Delete (CRUD) operations with real-time input validation and CSV persistence.
2. Implement robust **Attendance and Academic Risk Monitoring** based on transparent educational thresholds (Attendance < 75%, Final Score < 50, or Backlogs ≥ 2).
3. Conduct in-depth **Data Science Analytics** utilizing Pandas and NumPy for data inspection, cleaning, descriptive statistics (mean, median, std, skewness), and correlation heatmaps.
4. Deploy an **Explainable Supervised Machine Learning Classifier** (Decision Tree) to predict student performance categories with feature importance metrics.
5. Engineer a beyond-syllabus **Autonomous Agentic Data Science Workflow Controller** capable of data understanding, quality auditing, strategic analytical decision-making, and dynamic insight generation.
6. Provide an intuitive, responsive **Streamlit User Interface** designed for a rapid 5–10 minute viva demonstration.

---

## 6. Existing System
Existing academic information systems are typically centralized relational databases (e.g., legacy ERPs) that focus strictly on administrative transactional operations (e.g., fee payment, enrollment).
- **Disadvantages of Existing Systems**:
  - Complex database and network requirements.
  - Lack of built-in data science exploratory data analysis (EDA).
  - No autonomous intelligence to detect anomalies or suggest interventions.
  - Heavy monolithic deployment structures unsuitable for rapid academic prototyping or classroom demonstration.

---

## 7. Proposed System
The proposed system combines an intuitive Streamlit front-end with a modular Python backend:
- **Portability**: Operates directly on structured CSV datasets without external database servers.
- **Integrated Analytics**: Integrates descriptive statistics, group-by breakdowns, and Seaborn visualizations directly into the UI.
- **Early Warning System**: Flags at-risk students in real-time as grades or attendance values change.
- **Autonomous Agentic Workflow**: Executes a multi-stage autonomous pipeline that evaluates data health, selects features, trains predictive models, and derives evidence-backed insights automatically.

---

## 8. Technology Stack
- **Programming Language**: Python 3.14 / 3.x
- **User Interface Framework**: Streamlit (v1.64+)
- **Data Manipulation**: Pandas (v3.0+)
- **Numerical Computation**: NumPy (v2.5+)
- **Data Visualization**: Matplotlib (v3.11+) and Seaborn (v0.13+)
- **Machine Learning**: Scikit-learn (v1.9+)
- **Data Storage**: CSV (Comma-Separated Values) format with thread-safe file handling

---

## 9. Dataset Description
The dataset contains **400 student records** representing undergraduate students across five engineering branches:
- `StudentID`: Unique identifier (`STU1001` to `STU1400`)
- `Name`: Full name of the student
- `Age`: Age ranging between 18 and 24 years
- `Gender`: Categorical (`Male`, `Female`)
- `Department`: Categorical (`Computer Science`, `Information Technology`, `Electronics & Comm`, `Mechanical Eng`, `Civil Eng`)
- `Semester`: Integer (1 to 8)
- `Attendance`: Continuous percentage (0.0% to 100.0%)
- `StudyHoursPerWeek`: Weekly self-study hours (4.0 to 38.0 hours)
- `AssignmentScore`: Continuous assessment score (0 to 100)
- `MidtermScore`: Continuous midterm exam score (0 to 100)
- `FinalScore`: Derived composite academic performance (0 to 100)
- `PreviousSemesterScore`: Prior semester GPA equivalent (0 to 100)
- `Backlogs`: Count of active backlogs (0 to 6)
- `ExtracurricularHours`: Weekly hours devoted to sports/clubs (1.0 to 15.0)
- `PerformanceCategory`: Categorical (`High Performer`, `Good Performer`, `Average Performer`, `Needs Improvement`)
- `RiskStatus`: Categorical (`Safe`, `At Risk`)

---

## 10. Dataset Generation
The dataset was synthetically generated using `generate_dataset.py` with reproducible random seeds (`seed=42`). Rather than assigning values randomly, mathematically coherent inter-variable correlations were modeled:
- Attendance is modeled via a bimodal distribution: ~78% regular attendees ($\mu = 83.5\%$) and ~22% irregular attendees ($\mu = 61.0\%$).
- Assignment and Midterm scores are positively correlated with attendance and weekly study hours.
- Backlogs are generated via Poisson processes inversely proportional to midterm scores and attendance.
- Final Score is derived using the objective formula:
  $$\text{FinalScore} = 0.25 \times \text{Assignment} + 0.35 \times \text{Midterm} + 0.20 \times (\text{Attendance} \times 0.85) + 0.15 \times (\text{StudyHours} \times 2.4) - (\text{Backlogs} \times 2.5)$$

---

## 11. System Architecture
```text
┌────────────────────────────────────────────────────────┐
│                   Streamlit Web GUI                    │
│   (Dashboard | Students | Attendance | Marks | DS | ML)│
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌────────────────────┐     ┌───────────────────────┐
   │  student_manager   │     │  agent.py (Agentic)   │
   │  (CRUD & Rules)    │     │  (Workflow Controller)│
   └──────────┬─────────┘     └───────────┬───────────┘
              │                           │
              │  ┌────────────────────────┴────────┐
              ▼  ▼                                 ▼
   ┌────────────────────┐                ┌──────────────────┐
   │ data_analysis.py   │                │   ml_model.py    │
   │ (Pandas / NumPy)   │                │  (Scikit-Learn)  │
   └──────────┬─────────┘                └─────────┬────────┘
              │                                    │
              ▼                                    ▼
   ┌────────────────────┐                ┌──────────────────┐
   │  visualization.py  │                │ data/students.csv│
   │ (Matplotlib/Seaborn│                │  (Persistent DB) │
   └────────────────────┘                └──────────────────┘
```

---

## 12. Student Management Module
Implements complete CRUD capabilities:
- **Create**: Add new student with exhaustive validation (age 16–45, semester 1–8, attendance 0–100%, scores 0–100%, non-negative backlogs).
- **Read / Search**: Real-time multi-attribute filtering by ID, Name substring, Department, Category, and Risk Status.
- **Update**: Select existing student ID, populate current values into input fields, update records, and automatically recompute derived scores and risk statuses.
- **Delete**: Remove student with explicit confirmation to prevent data loss.

---

## 13. Data Science Module
Implements standard Python for Data Science curriculum operations:
- **Data Inspection**: Automated audit of column data types, missing null values, and memory footprints.
- **Data Cleaning**: Median imputation for numerical features, mode imputation for categorical attributes, and boundary clipping.
- **Descriptive Statistics**: Computing Mean, Median, Standard Deviation, Min, Q1, Q2, Q3, Max, and Skewness using Pandas and NumPy.
- **Correlation Analysis**: Generating Pearson correlation matrices and extracting top positively and negatively correlated feature pairs.
- **Group-By Aggregations**: Computing performance metrics stratified across departments, semesters, attendance brackets (<65%, 65–75%, 75–85%, >85%), and weekly study hour brackets.

---

## 14. Machine Learning Module
Predicts student `PerformanceCategory` using an explainable **Decision Tree Classifier** (default `max_depth=4`):
- **Features Used**: `Attendance`, `StudyHoursPerWeek`, `AssignmentScore`, `MidtermScore`, `PreviousSemesterScore`, `Backlogs`.
- **Target**: `PerformanceCategory` (4 classes: High Performer, Good Performer, Average Performer, Needs Improvement).
- **Evaluation**: 75% Training / 25% Stratified Holdout Test split.
- **Explainability**: Outputs feature importance percentages and confusion matrix heatmaps.
- **What-If Simulation**: An interactive GUI allowing faculty to manipulate slider attributes and immediately observe predicted category probabilities.

---

## 15. Agentic Data Science Module (Beyond-Syllabus Novelty)
The defining innovation of this project is the **Autonomous Agentic Workflow Controller** (`agent.py`). Rather than running static, hard-coded scripts:
1. The agent acts as an autonomous data scientist.
2. It executes a step-by-step decision Directed Acyclic Graph (DAG).
3. At each stage, the agent logs its internal reasoning ("Agent Thought"), the action executed, the findings uncovered, and the subsequent branching decision taken.

---

## 16. Autonomous Workflow Execution
```text
Dataset Received
       │
       ▼
[Stage 1: Data Understanding Agent]
- Infers cardinality, numeric/categorical split, memory profile
       │
       ▼
[Stage 2: Data Quality & Anomaly Audit]
- Audits null percentages and duplicate rows
       │
   Is Quality Breached?
    ├── YES ──► [Stage 3: Autonomous Cleansing] (Median/Mode imputation, deduplication)
    └── NO  ──► [Stage 3: Standard Boundary Clipping Guard]
       │
       ▼
[Stage 4: Strategic Analysis Decision Engine]
- Evaluates sample size (N ≥ 50) for ML viability
- Ranks top Pearson correlations to determine EDA focus
       │
       ▼
[Stage 5: Autonomous Targeted EDA Execution]
- Computes multi-bracket aggregations and statistical profiles
       │
       ▼
[Stage 6: Autonomous ML Training & Holdout Evaluation]
- Fits Decision Tree, evaluates generalization, identifies top driver
       │
       ▼
[Stage 7: Dynamic Evidence-Based Insight Synthesis]
- Formulates mathematical statements and recommendations directly from data
       │
       ▼
[Stage 8: Executive Summary & DAG Trace Generation]
```

---

## 17. Algorithms & Mathematical Methods
1. **Pearson Product-Moment Correlation Coefficient**:
   $$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
2. **Decision Tree Classification (CART Algorithm)**:
   Splits nodes by minimizing Gini Impurity:
   $$I_G(p) = 1 - \sum_{i=1}^{J} p_i^2$$
3. **Student Risk Rule**:
   $$\text{RiskStatus} = \begin{cases} \text{At Risk}, & \text{if } \text{Attendance} < 75\% \lor \text{FinalScore} < 50 \lor \text{Backlogs} \ge 2 \\ \text{Safe}, & \text{otherwise} \end{cases}$$

---

## 18. Application Screenshots & Interface Overview
- **Dashboard View**: Six prominent KPI metric cards, Departmental Final Score bar chart, Performance Category donut chart, Attendance histogram, and Risk Status distribution.
- **Student Management View**: Four interactive tabs for searching, registering, modifying, and deleting student rows.
- **Attendance Monitor**: Dynamic threshold slider linked in real-time to alert tables and KDE distribution plots.
- **Analytics View**: Comprehensive Pandas/NumPy statistical tables, Pearson heatmap, and cohort breakdowns.
- **ML Classifier View**: Train/test accuracy displays, feature importance bar charts, confusion matrices, and live interactive slider prediction.
- **Agentic Analysis View**: One-click workflow execution with step-by-step DAG trace, agent thought bubbles, and dynamic insight cards.

---

## 19. Core Code Implementation Snippets

### A. Rule-Based Score & Risk Derivations (`student_manager.py`)
```python
def derive_final_score(assignment, midterm, attendance, study_hours, backlogs):
    raw_score = (
        0.25 * float(assignment) +
        0.35 * float(midterm) +
        0.20 * (float(attendance) * 0.85) +
        0.15 * (float(study_hours) * 2.4) -
        (int(backlogs) * 2.5)
    )
    return round(float(np.clip(raw_score, 0.0, 100.0)), 1)
```

### B. Autonomous Decision Engine (`agent.py`)
```python
sample_size = len(cleaned_df)
can_run_ml = sample_size >= 50 and "PerformanceCategory" in cleaned_df.columns
corr_matrix, top_pairs = da.compute_correlation_matrix(cleaned_df)
primary_driver = top_pairs[0] if top_pairs else None
```

---

## 20. Results & Findings
- **Sample Distribution**: In the 400-student cohort, 48.0% were classified as Average Performers, 40.0% Good Performers, 9.3% Needs Improvement, and 2.8% High Performers.
- **Attendance Impact**: Students maintaining $\ge 75\%$ attendance averaged significantly higher final scores than peers with $< 75\%$ attendance.
- **Study Hours Correlation**: Weekly study hours exhibited strong positive correlation ($r \approx 0.65$) with academic scores.
- **ML Accuracy**: The explainable Decision Tree classifier achieved $> 85\%$ accuracy on the holdout test dataset, identifying Midterm Scores and Attendance as primary split drivers.
- **Risk Identification**: 124 students (31.0%) were flagged for early educational intervention.

---

## 21. Advantages of Proposed System
1. **Lightweight & Portable**: Runs entirely on CSV and local Python without requiring SQL database servers, Docker, or external API keys.
2. **Pedagogically Explainable**: Every prediction and analytical step can be transparently justified during college viva evaluations.
3. **Beyond-Syllabus Innovation**: Demonstrates actual agentic decision loops rather than superficial static text.
4. **Data Integrity**: Enforces strict input validation and non-destructive data updates.

---

## 22. Limitations
1. The synthetic dataset represents a simulated engineering college environment and may not reflect unique real-world external disruptions (e.g., medical leaves).
2. The current implementation utilizes local file persistence; multi-user concurrent write operations would require database locking mechanisms in enterprise deployments.

---

## 23. Future Scope
1. Integrate Large Language Model (LLM) agents via local Ollama or Gemini APIs for conversational natural-language querying of student records.
2. Extend the system to automate personalized email and SMS remediation notices to parents of at-risk students.
3. Incorporate time-series tracking across multiple semesters to predict longitudinal graduation trajectories.

---

## 24. Conclusion
The **Agentic Student Management and Data Science Analytics System** fulfills all requirements of a Project-Based Learning (PBL) curriculum for Python for Data Science. It bridges conventional record management with exploratory data analysis and supervised machine learning, while demonstrating advanced autonomous workflow orchestration. The project provides an ideal balance of functional software engineering, statistical rigor, and high-scoring viva demonstration capabilities.

---

## 25. References
1. McKinney, W. (2022). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter* (3rd ed.). O'Reilly Media.
2. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
3. Streamlit Documentation. *Streamlit: A faster way to build and share data apps*. https://docs.streamlit.io/
4. Scikit-learn Developers. *Supervised learning: Decision Trees*. https://scikit-learn.org/stable/modules/tree.html
5. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
