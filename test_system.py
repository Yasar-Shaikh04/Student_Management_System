"""
test_system.py
--------------
Automated end-to-end test suite for the Agentic Student Management System.
Validates dataset integrity, CRUD operations, Data Science routines,
visualization generators, ML training/prediction, and the Autonomous Agent workflow.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

import generate_dataset as gd
import student_manager as sm
import data_analysis as da
import visualization as viz
import ml_model as mlm
import agent as ag

class TestAgenticStudentManagementSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv = "data/test_students.csv"
        # Generate a test dataset of 100 records
        df = gd.generate_students(n_records=100, random_seed=99)
        os.makedirs(os.path.dirname(cls.test_csv), exist_ok=True)
        df.to_csv(cls.test_csv, index=False)
        cls.df = sm.load_students_data(cls.test_csv)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv):
            try:
                os.remove(cls.test_csv)
            except Exception:
                pass

    def test_01_dataset_generation(self):
        """Verify generated dataset dimensions, columns, and data types."""
        self.assertEqual(len(self.df), 100)
        expected_cols = [
            "StudentID", "Name", "Age", "Gender", "Department", "Semester",
            "Attendance", "StudyHoursPerWeek", "AssignmentScore", "MidtermScore",
            "FinalScore", "PreviousSemesterScore", "Backlogs", "ExtracurricularHours",
            "PerformanceCategory", "RiskStatus"
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)
        self.assertEqual(self.df["StudentID"].nunique(), 100)
        self.assertTrue((self.df["Attendance"] >= 0).all() and (self.df["Attendance"] <= 100).all())
        self.assertTrue((self.df["FinalScore"] >= 0).all() and (self.df["FinalScore"] <= 100).all())

    def test_02_student_manager_crud(self):
        """Test Add, Search, Update, and Delete operations."""
        df_work = self.df.copy()
        
        # Test Add Student
        new_stu = {
            "StudentID": "STU9999",
            "Name": "Test Student",
            "Age": 21,
            "Gender": "Male",
            "Department": "Computer Science",
            "Semester": 5,
            "Attendance": 55.0,  # Low attendance -> should be At Risk
            "StudyHoursPerWeek": 10.0,
            "AssignmentScore": 50.0,
            "MidtermScore": 48.0,
            "PreviousSemesterScore": 52.0,
            "Backlogs": 2,
            "ExtracurricularHours": 2.0
        }
        success, msg, df_work = sm.add_student(df_work, new_stu, csv_path=self.test_csv)
        self.assertTrue(success)
        self.assertEqual(len(df_work), 101)
        
        added_row = df_work[df_work["StudentID"] == "STU9999"].iloc[0]
        self.assertEqual(added_row["RiskStatus"], "At Risk")
        
        # Test Search
        search_res = sm.search_students(df_work, query="Test Student")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res.iloc[0]["StudentID"], "STU9999")
        
        # Test Update Student
        update_data = {
            "StudentID": "STU9999",
            "Attendance": 92.0,
            "MidtermScore": 88.0,
            "AssignmentScore": 90.0,
            "Backlogs": 0
        }
        success_up, msg_up, df_work = sm.update_student(df_work, "STU9999", update_data, csv_path=self.test_csv)
        self.assertTrue(success_up)
        
        updated_row = df_work[df_work["StudentID"] == "STU9999"].iloc[0]
        self.assertEqual(updated_row["Attendance"], 92.0)
        self.assertEqual(updated_row["RiskStatus"], "Safe")
        
        # Test Delete Student
        success_del, msg_del, df_work = sm.delete_student(df_work, "STU9999", csv_path=self.test_csv)
        self.assertTrue(success_del)
        self.assertEqual(len(df_work), 100)
        self.assertEqual(len(df_work[df_work["StudentID"] == "STU9999"]), 0)

    def test_03_data_analysis(self):
        """Test Pandas & NumPy statistical profiling, cleaning, correlations, and group-by."""
        inspect = da.inspect_dataset(self.df)
        self.assertEqual(inspect["total_rows"], 100)
        self.assertTrue(len(inspect["numeric_columns"]) >= 7)
        
        # Test missing & duplicates check
        missing_df = da.check_missing_values(self.df)
        self.assertEqual(len(missing_df), 0)
        
        dup_count, _ = da.check_duplicate_records(self.df)
        self.assertEqual(dup_count, 0)
        
        # Test descriptive stats
        stats_df = da.calculate_descriptive_statistics(self.df)
        self.assertTrue(len(stats_df) >= 7)
        self.assertIn("Mean", stats_df.columns)
        self.assertIn("Skewness", stats_df.columns)
        
        # Test correlation matrix
        corr_matrix, top_pairs = da.compute_correlation_matrix(self.df)
        self.assertGreater(len(top_pairs), 0)
        self.assertTrue(-1.0 <= top_pairs[0]["Correlation"] <= 1.0)
        
        # Test group-by aggregations
        grp = da.perform_group_by_analysis(self.df)
        self.assertIn("Department_Analysis", grp)
        self.assertIn("Attendance_Bracket_Analysis", grp)
        self.assertIn("Study_Bracket_Analysis", grp)

    def test_04_visualizations(self):
        """Verify that all visualization routines return valid matplotlib figures."""
        fig1 = viz.plot_attendance_distribution(self.df)
        self.assertIsNotNone(fig1)
        
        fig2 = viz.plot_marks_distribution(self.df)
        self.assertIsNotNone(fig2)
        
        fig3 = viz.plot_department_performance(self.df)
        self.assertIsNotNone(fig3)
        
        fig4 = viz.plot_attendance_vs_marks(self.df)
        self.assertIsNotNone(fig4)
        
        fig5 = viz.plot_study_hours_vs_performance(self.df)
        self.assertIsNotNone(fig5)
        
        fig6 = viz.plot_performance_distribution(self.df)
        self.assertIsNotNone(fig6)
        
        fig7 = viz.plot_correlation_heatmap(self.df)
        self.assertIsNotNone(fig7)
        
        fig8 = viz.plot_risk_distribution(self.df)
        self.assertIsNotNone(fig8)
        
        # Test saving chart
        saved_path = viz.save_fig_to_disk(fig1, "test_chart.png")
        self.assertTrue(os.path.exists(saved_path))
        if os.path.exists(saved_path):
            os.remove(saved_path)

    def test_05_machine_learning(self):
        """Test Decision Tree training, accuracy evaluation, feature importance, and interactive prediction."""
        predictor = mlm.StudentPerformancePredictor(model_type="Decision Tree", max_depth=4)
        metrics = predictor.train(self.df)
        
        self.assertTrue(predictor.is_trained)
        self.assertGreater(metrics["train_accuracy"], 50.0)
        self.assertGreater(metrics["test_accuracy"], 50.0)
        self.assertGreater(len(metrics["feature_importances"]), 0)
        
        # Test what-if prediction on single student
        sample_input = {
            "Attendance": 88.0,
            "StudyHoursPerWeek": 22.0,
            "AssignmentScore": 85.0,
            "MidtermScore": 82.0,
            "PreviousSemesterScore": 80.0,
            "Backlogs": 0
        }
        pred_res = predictor.predict_single(sample_input)
        self.assertIn("prediction", pred_res)
        self.assertIn(pred_res["prediction"], mlm.CATEGORY_LABELS)
        self.assertGreater(pred_res["confidence"], 0.0)

    def test_06_agentic_workflow(self):
        """Test the autonomous Agentic Data Science controller and dynamic insight generation."""
        controller = ag.AgenticWorkflowController()
        results = controller.run_pipeline(self.df)
        
        # Verify trace execution
        self.assertGreater(len(results["trace"]), 5)
        step_stages = [s.stage for s in results["trace"]]
        self.assertTrue(any("Data Ingestion & Understanding" in s for s in step_stages))
        self.assertTrue(any("Data Quality & Anomaly Audit" in s for s in step_stages))
        self.assertTrue(any("Strategic Analysis Decision Engine" in s for s in step_stages))
        self.assertTrue(any("Autonomous Dynamic Insight Synthesis" in s for s in step_stages))
        
        # Verify dynamic insights generated
        self.assertGreater(len(results["insights"]), 3)
        for ins in results["insights"]:
            self.assertIn("title", ins)
            self.assertIn("detail", ins)
            self.assertIn("recommendation", ins)
            self.assertGreater(len(ins["detail"]), 20)
            
        # Verify markdown summary generation
        summary = controller.generate_markdown_summary()
        self.assertIn("Autonomous Agentic Data Science Executive Summary", summary)
        self.assertIn("Autonomous Decision Trace", summary)

if __name__ == "__main__":
    unittest.main()
