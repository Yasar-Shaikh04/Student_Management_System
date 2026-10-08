"""
ml_model.py
------------
Machine Learning module predicting student 'Performance Category'
using Scikit-learn models (Decision Tree Classifier & Random Forest Classifier).
Provides model training, evaluation, feature importance, confusion matrix,
and an interactive what-if prediction interface.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "Attendance",
    "StudyHoursPerWeek",
    "AssignmentScore",
    "MidtermScore",
    "PreviousSemesterScore",
    "Backlogs"
]

TARGET_COLUMN = "PerformanceCategory"

CATEGORY_LABELS = [
    "High Performer",
    "Good Performer",
    "Average Performer",
    "Needs Improvement"
]

class StudentPerformancePredictor:
    def __init__(self, model_type: str = "Decision Tree", max_depth: int = 4):
        self.model_type = model_type
        self.max_depth = max_depth
        if model_type == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=50, max_depth=max_depth, random_state=42)
        else:
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            
        self.is_trained = False
        self.feature_names = FEATURE_COLUMNS
        self.classes_ = []
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.classification_rep = ""
        self.conf_matrix = None
        self.feature_importances_ = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepares feature matrix X and target y, and splits into train/test sets."""
        df_clean = df.dropna(subset=self.feature_names + [TARGET_COLUMN]).copy()
        X = np.asarray(df_clean[self.feature_names].values, dtype=np.float64)
        y = np.asarray(df_clean[TARGET_COLUMN].tolist(), dtype=str)
        
        # Check if all classes have at least 2 members for stratification
        _, counts = np.unique(y, return_counts=True)
        stratify_arg = y if (len(counts) > 1 and np.min(counts) >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify_arg
        )
        return X_train, X_test, y_train, y_test

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains the classification model and evaluates performance metrics."""
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        self.model.fit(X_train, y_train)
        self.classes_ = list(self.model.classes_)
        
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        self.train_accuracy = round(float(accuracy_score(y_train, y_train_pred)) * 100, 2)
        self.test_accuracy = round(float(accuracy_score(y_test, y_test_pred)) * 100, 2)
        
        self.classification_rep = classification_report(y_test, y_test_pred, zero_division=0)
        self.conf_matrix = confusion_matrix(y_test, y_test_pred, labels=self.classes_)
        
        # Calculate feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self.feature_importances_ = {
                name: round(float(imp), 4)
                for name, imp in zip(self.feature_names, importances)
            }
            
        self.is_trained = True
        
        return {
            "model_type": self.model_type,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "feature_importances": self.feature_importances_,
            "classes": self.classes_
        }

    def predict_single(self, input_features: Dict[str, float]) -> Dict[str, Any]:
        """Predicts performance category and class probabilities for a single student."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating predictions.")
            
        x_vec = np.array([[float(input_features[f]) for f in self.feature_names]])
        pred_label = self.model.predict(x_vec)[0]
        
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_vec)[0]
            for cls_name, p in zip(self.classes_, probs):
                probabilities[cls_name] = round(float(p) * 100, 1)
                
        # Confidence score
        confidence = probabilities.get(pred_label, 100.0)
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "input_features": input_features
        }

    def plot_feature_importance(self) -> plt.Figure:
        """Returns a bar chart showing relative feature importances."""
        fig, ax = plt.subplots(figsize=(7.5, 4))
        if not self.feature_importances_:
            ax.text(0.5, 0.5, "Feature importances not available", ha="center")
            return fig
            
        sorted_fi = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_fi]
        vals = [x[1] * 100 for x in sorted_fi]
        
        colors = sns.color_palette("viridis", len(names))
        bars = ax.barh(names[::-1], vals[::-1], color=colors, height=0.55)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, weight="bold")
            
        ax.set_title(f"Feature Importance ({self.model_type})", fontsize=12, weight="bold", pad=10)
        ax.set_xlabel("Relative Importance (%)", fontsize=10)
        ax.set_xlim(0, max(vals) * 1.25 if vals else 100)
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self) -> plt.Figure:
        """Plots confusion matrix heatmap for test evaluation."""
        fig, ax = plt.subplots(figsize=(6.5, 5))
        if self.conf_matrix is None:
            ax.text(0.5, 0.5, "Model not trained yet", ha="center")
            return fig
            
        sns.heatmap(
            self.conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            cbar=False,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix ({self.model_type})", fontsize=12, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        return fig

def train_and_cache_model(df: pd.DataFrame, model_type: str = "Decision Tree", max_depth: int = 4) -> StudentPerformancePredictor:
    """Helper function to instantiate and train a predictor."""
    predictor = StudentPerformancePredictor(model_type=model_type, max_depth=max_depth)
    predictor.train(df)
    return predictor
