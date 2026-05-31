"""
test_prediction.py - Unit tests for prediction and recommendation engines.
Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _train_and_get_predictor():
    """Train a fresh model and return a StudentPredictor using it."""
    from config.config           import DEPARTMENTS
    from src.data_preprocessing  import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from src.train_model         import ModelTrainer
    from src.predict             import StudentPredictor

    rng = np.random.default_rng(7)
    n   = 150
    df  = pd.DataFrame({
        "student_id":               [f"TST{i:04d}" for i in range(n)],
        "name":                     [f"S{i}" for i in range(n)],
        "gender":                   rng.choice(["Male", "Female"], n),
        "age":                      rng.integers(18, 24, n),
        "department":               rng.choice(DEPARTMENTS[:2], n),
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

    dp = DataPreprocessor()
    df = dp.run(df)
    fe = FeatureEngineer()
    df = fe.run(df, save=False)

    trainer = ModelTrainer()
    trainer.run(df)

    return StudentPredictor(trainer.best_model_path if hasattr(trainer, "best_model_path")
                            else None)


SAMPLE_RECORD = {
    "student_id":               "TST9999",
    "name":                     "Test Student",
    "gender":                   "Male",
    "age":                      20,
    "department":               "Computer Science Engineering",
    "attendance_percentage":    55.0,
    "assignment_score":         42.0,
    "quiz_score":               38.0,
    "lab_score":                48.0,
    "midterm_marks":            35.0,
    "lms_activity":             25.0,
    "videos_watched":           4,
    "discussion_participation": 2,
}

GOOD_RECORD = {
    "student_id":               "TST0001",
    "name":                     "Good Student",
    "gender":                   "Female",
    "age":                      19,
    "department":               "Computer Science Engineering",
    "attendance_percentage":    95.0,
    "assignment_score":         90.0,
    "quiz_score":               88.0,
    "lab_score":                92.0,
    "midterm_marks":            85.0,
    "lms_activity":             90.0,
    "videos_watched":           45,
    "discussion_participation": 30,
}


class TestStudentPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.predict import StudentPredictor
        from config.config import BEST_MODEL_PATH
        # Use whichever model file is available from prior test runs
        cls.predictor = StudentPredictor()

    def _safe_predict(self, record):
        try:
            return self.predictor.predict_single(record)
        except FileNotFoundError:
            self.skipTest("Model file not found — run training tests first.")

    def test_single_prediction_returns_expected_keys(self):
        result = self._safe_predict(SAMPLE_RECORD)
        for key in ("predicted_result", "risk_level", "risk_score",
                    "pass_probability", "fail_probability", "confidence"):
            self.assertIn(key, result)

    def test_predicted_result_is_pass_or_fail(self):
        result = self._safe_predict(SAMPLE_RECORD)
        self.assertIn(result["predicted_result"], {"Pass", "Fail"})

    def test_risk_level_is_valid(self):
        result = self._safe_predict(SAMPLE_RECORD)
        self.assertIn(result["risk_level"], {"High Risk", "Medium Risk", "Low Risk"})

    def test_probabilities_sum_to_one(self):
        result = self._safe_predict(SAMPLE_RECORD)
        total = result["pass_probability"] + result["fail_probability"]
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_risk_score_in_range(self):
        result = self._safe_predict(SAMPLE_RECORD)
        self.assertGreaterEqual(result["risk_score"], 0.0)
        self.assertLessEqual   (result["risk_score"], 100.0)

    def test_confidence_in_range(self):
        result = self._safe_predict(SAMPLE_RECORD)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual   (result["confidence"], 100.0)

    def test_good_student_lower_risk(self):
        """A high-scoring student should have lower risk than a low-scoring one."""
        r_bad  = self._safe_predict(SAMPLE_RECORD)
        r_good = self._safe_predict(GOOD_RECORD)
        self.assertLessEqual(r_good["risk_score"], r_bad["risk_score"])

    def test_batch_prediction(self):
        df = pd.DataFrame([SAMPLE_RECORD, GOOD_RECORD])
        try:
            df_result = self.predictor.predict_batch(df)
        except FileNotFoundError:
            self.skipTest("Model file not found.")
        self.assertEqual(len(df_result), 2)
        self.assertIn("predicted_result", df_result.columns)
        self.assertIn("risk_level",       df_result.columns)


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        from src.recommendation_engine import RecommendationEngine
        self.engine = RecommendationEngine()

    def test_returns_list(self):
        recs = self.engine.generate(SAMPLE_RECORD, "High Risk")
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)

    def test_recommendation_has_required_keys(self):
        recs = self.engine.generate(SAMPLE_RECORD, "High Risk")
        for r in recs:
            for key in ("category", "priority", "icon", "message", "action"):
                self.assertIn(key, r)

    def test_high_risk_includes_critical(self):
        recs = self.engine.generate(SAMPLE_RECORD, "High Risk")
        priorities = {r["priority"] for r in recs}
        self.assertIn("CRITICAL", priorities)

    def test_low_risk_good_student(self):
        recs = self.engine.generate(GOOD_RECORD, "Low Risk")
        self.assertIsInstance(recs, list)
        # Good student should have at least one rec (the positive one)
        self.assertGreater(len(recs), 0)

    def test_format_for_display(self):
        recs    = self.engine.generate(SAMPLE_RECORD, "High Risk")
        display = self.engine.format_for_display(recs)
        self.assertIsInstance(display, list)
        for line in display:
            self.assertIsInstance(line, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
