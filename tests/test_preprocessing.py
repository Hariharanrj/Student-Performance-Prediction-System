"""
test_preprocessing.py - Unit tests for data preprocessing pipeline.
Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing  import DataPreprocessor
from src.feature_engineering import FeatureEngineer


def _make_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal valid student DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "student_id":               [f"STU{i:04d}" for i in range(n)],
        "name":                     [f"Student {i}" for i in range(n)],
        "gender":                   rng.choice(["Male", "Female", "Other"], n),
        "age":                      rng.integers(18, 24, n),
        "department":               rng.choice(["CS", "IT", "ECE"], n),
        "attendance_percentage":    rng.uniform(40, 100, n),
        "assignment_score":         rng.uniform(20, 100, n),
        "quiz_score":               rng.uniform(20, 100, n),
        "lab_score":                rng.uniform(20, 100, n),
        "midterm_marks":            rng.uniform(20, 100, n),
        "lms_activity":             rng.uniform(10, 100, n),
        "videos_watched":           rng.integers(0, 40, n),
        "discussion_participation": rng.integers(0, 30, n),
        "result":                   rng.choice(["Pass", "Fail"], n),
    })


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.dp = DataPreprocessor()
        self.df = _make_df(20)

    def test_run_returns_dataframe(self):
        out = self.dp.run(self.df)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertGreater(len(out), 0)

    def test_no_missing_values_after_run(self):
        out = self.dp.run(self.df)
        numeric_cols = ["attendance_percentage", "assignment_score",
                        "quiz_score", "lab_score", "midterm_marks"]
        for col in numeric_cols:
            if col in out.columns:
                self.assertEqual(out[col].isna().sum(), 0,
                                 f"NaN found in {col} after preprocessing")

    def test_target_encoding(self):
        out = self.dp.run(self.df)
        self.assertIn("result", out.columns)
        self.assertTrue(set(out["result"].unique()).issubset({0, 1}),
                        "result column should only contain 0 and 1")

    def test_duplicates_removed(self):
        df_dup = pd.concat([self.df, self.df.head(3)], ignore_index=True)
        out = self.dp.run(df_dup)
        self.assertEqual(len(out), len(self.df))

    def test_outlier_capping(self):
        df_outlier = self.df.copy()
        df_outlier.loc[0, "attendance_percentage"] = 150.0  # above max
        df_outlier.loc[1, "attendance_percentage"] = -10.0  # below min
        out = self.dp.run(df_outlier)
        self.assertLessEqual(out["attendance_percentage"].max(), 100.0)
        self.assertGreaterEqual(out["attendance_percentage"].min(), 0.0)

    def test_column_standardisation(self):
        df_messy = self.df.rename(columns={"student_id": "Student ID"})
        df_messy.columns = [c.upper() for c in df_messy.columns]
        # Just check it runs without error after standardisation
        out = self.dp.run(df_messy)
        self.assertIsNotNone(out)


class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        dp = DataPreprocessor()
        df = _make_df(30)
        self.df_processed = dp.run(df)
        self.fe = FeatureEngineer()

    def test_run_adds_engineered_features(self):
        from config.config import ENGINEERED_FEATURES
        out = self.fe.run(self.df_processed, save=False)
        for feat in ENGINEERED_FEATURES:
            self.assertIn(feat, out.columns, f"Engineered feature '{feat}' missing")

    def test_attendance_risk_values(self):
        out = self.fe.run(self.df_processed, save=False)
        self.assertTrue(out["attendance_risk"].isin([0, 1, 2]).all(),
                        "attendance_risk must be 0, 1, or 2")

    def test_assignment_ratio_in_bounds(self):
        out = self.fe.run(self.df_processed, save=False)
        self.assertTrue((out["assignment_completion_ratio"] >= 0).all())
        self.assertTrue((out["assignment_completion_ratio"] <= 1).all())

    def test_engagement_score_in_bounds(self):
        out = self.fe.run(self.df_processed, save=False)
        self.assertTrue((out["engagement_score"] >= 0).all())
        self.assertTrue((out["engagement_score"] <= 100).all())

    def test_overall_score_in_bounds(self):
        out = self.fe.run(self.df_processed, save=False)
        self.assertTrue((out["overall_academic_score"] >= 0).all())
        self.assertTrue((out["overall_academic_score"] <= 100).all())

    def test_no_nan_in_engineered_features(self):
        from config.config import ENGINEERED_FEATURES
        out = self.fe.run(self.df_processed, save=False)
        for feat in ENGINEERED_FEATURES:
            if feat in out.columns:
                self.assertEqual(out[feat].isna().sum(), 0,
                                 f"NaN found in engineered feature: {feat}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
