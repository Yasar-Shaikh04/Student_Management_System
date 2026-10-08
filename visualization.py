"""
visualization.py
----------------
Generates clean, aesthetic, academic-ready charts using Matplotlib and Seaborn.
Functions return matplotlib.figure.Figure objects suitable for Streamlit (st.pyplot).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set cohesive academic aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "charts")

def save_fig_to_disk(fig: plt.Figure, filename: str) -> str:
    """Helper to save figure to disk for report export."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    return filepath

def plot_attendance_distribution(df: pd.DataFrame, threshold: float = 75.0) -> plt.Figure:
    """Plots attendance histogram with KDE and highlights the low-attendance danger threshold."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    sns.histplot(
        df["Attendance"],
        kde=True,
        bins=20,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.75,
        ax=ax
    )
    ax.axvline(
        threshold,
        color="#d9534f",
        linestyle="--",
        linewidth=2,
        label=f"Warning Threshold ({threshold}%)"
    )
    
    low_att_pct = (df["Attendance"] < threshold).mean() * 100
    ax.set_title(f"Student Attendance Distribution ({low_att_pct:.1f}% Below Threshold)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Attendance Percentage (%)", fontsize=11)
    ax.set_ylabel("Student Count", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig

def plot_marks_distribution(df: pd.DataFrame) -> plt.Figure:
    """Visualizes comparative distributions of Assignment, Midterm, and Final Scores."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    palette = ["#3498db", "#e67e22", "#2ecc71"]
    sns.kdeplot(df["AssignmentScore"], fill=True, label="Assignment Score", color=palette[0], alpha=0.3, ax=ax)
    sns.kdeplot(df["MidtermScore"], fill=True, label="Midterm Score", color=palette[1], alpha=0.3, ax=ax)
    sns.kdeplot(df["FinalScore"], fill=True, label="Final Score", color=palette[2], alpha=0.4, linewidth=2, ax=ax)
    
    ax.set_title("Comparative Marks Distribution (Assignment vs Midterm vs Final)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Score (out of 100)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig

def plot_department_performance(df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing average Final Score across academic departments."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    
    dept_stats = df.groupby("Department")["FinalScore"].mean().reset_index().sort_values(by="FinalScore", ascending=True)
    
    colors = ["#4575b4", "#74add1", "#abd9e9", "#fdae61", "#f46d43"]
    bars = ax.barh(dept_stats["Department"], dept_stats["FinalScore"], color=colors[:len(dept_stats)], edgecolor="none", height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.8, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va="center", ha="left", fontsize=10, weight="bold")
        
    ax.set_xlim(0, 100)
    ax.set_title("Department-Wise Average Final Score", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Average Final Score", fontsize=11)
    ax.set_ylabel("Department", fontsize=11)
    plt.tight_layout()
    return fig

def plot_attendance_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot displaying Attendance vs Final Score with linear regression trendline."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    sns.regplot(
        data=df,
        x="Attendance",
        y="FinalScore",
        scatter_kws={"alpha": 0.6, "color": "#1f77b4", "s": 35},
        line_kws={"color": "#d62728", "linewidth": 2, "label": "Regression Trendline"},
        ax=ax
    )
    
    corr_val = df["Attendance"].corr(df["FinalScore"])
    ax.set_title(f"Attendance vs. Final Score (Pearson r = {corr_val:.2f})", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Attendance (%)", fontsize=11)
    ax.set_ylabel("Final Score", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig

def plot_study_hours_vs_performance(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot showing Study Hours vs Final Score colored by Performance Category."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    
    category_order = ["High Performer", "Good Performer", "Average Performer", "Needs Improvement"]
    palette = {"High Performer": "#2ca02c", "Good Performer": "#1f77b4", "Average Performer": "#ff7f0e", "Needs Improvement": "#d62728"}
    
    sns.scatterplot(
        data=df,
        x="StudyHoursPerWeek",
        y="FinalScore",
        hue="PerformanceCategory",
        hue_order=[c for c in category_order if c in df["PerformanceCategory"].unique()],
        palette=palette,
        alpha=0.8,
        s=50,
        ax=ax
    )
    
    ax.set_title("Weekly Study Hours vs. Final Score by Performance Category", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Weekly Study Hours", fontsize=11)
    ax.set_ylabel("Final Score", fontsize=11)
    ax.legend(title="Category", loc="lower right", framealpha=0.9)
    plt.tight_layout()
    return fig

def plot_performance_distribution(df: pd.DataFrame) -> plt.Figure:
    """Donut chart illustrating student distribution across Performance Categories."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    counts = df["PerformanceCategory"].value_counts()
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[:len(counts)],
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_weight("bold")
        
    ax.set_title("Performance Category Distribution", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Annotated Seaborn correlation heatmap across all key numerical attributes."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    numeric_cols = [
        "Attendance", "StudyHoursPerWeek", "AssignmentScore",
        "MidtermScore", "FinalScore", "PreviousSemesterScore", "Backlogs"
    ]
    present_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[present_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title("Academic Feature Correlation Matrix", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    return fig

def plot_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar plot representing the distribution of At Risk vs Safe students."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    risk_counts = df["RiskStatus"].value_counts()
    colors = {"Safe": "#28a745", "At Risk": "#dc3545"}
    bar_colors = [colors.get(x, "#6c757d") for x in risk_counts.index]
    
    bars = ax.bar(risk_counts.index, risk_counts.values, color=bar_colors, width=0.45)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 3, f"{int(y)} ({y/len(df)*100:.1f}%)", ha="center", weight="bold")
        
    ax.set_ylim(0, max(risk_counts.values) * 1.18)
    ax.set_title("Student Academic Risk Classification", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Number of Students", fontsize=11)
    plt.tight_layout()
    return fig
