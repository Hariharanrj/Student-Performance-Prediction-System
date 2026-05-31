"""
test_training.py - Unit tests for model training and evaluation.
Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_processed_df(n: int = 100) -> pd.DataFrame:
    """Generate a processed-like DataFrame with engineered features."""
    from src.data_preprocessing  import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from config.config           import DEPARTMENTS

    rng = np.random.default_rng(0)
    df  = pd.DataFrame({
        "student_id":               [f"STU{i:04d}" for i in range(n)],
        "name":                     [f"S{i}" for i in range(n)],
        "gender":                   rng.choice(["Male", "Female"], n),
        "age":                      rng.integers(18, 24, n),
        "department":               rng.choice(DEPARTMENTS[:3], n),
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

    dp      = DataPreprocessor()
    df_proc = dp.run(df)
    fe      = FeatureEngineer()
    return fe.run(df_proc, save=False)


class TestModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train once and share across tests."""
        from src.train_model import ModelTrainer
        cls.df_proc = _make_processed_df(200)
        cls.trainer = ModelTrainer()
        cls.results = cls.trainer.run(cls.df_proc)

    def test_results_not_empty(self):
        self.assertGreater(len(self.results), 0)

    def test_all_expected_models_trained(self):
        expected = {"Logistic Regression", "Random Forest"}   # XGBoost optional
        trained  = set(self.results.keys())
        self.assertTrue(expected.issubset(trained),
                        f"Expected {expected}, got {trained}")

    def test_metrics_in_valid_range(self):
        for name, m in self.results.items():
            with self.subTest(model=name):
                self.assertGreaterEqual(m["accuracy"],  0.0)
                self.assertLessEqual   (m["accuracy"],  1.0)
                self.assertGreaterEqual(m["f1"],        0.0)
                self.assertLessEqual   (m["f1"],        1.0)

    def test_best_model_is_set(self):
        self.assertIsNotNone(self.trainer.best_model)
        self.assertIn(self.trainer.best_name, self.results)

    def test_best_model_can_predict(self):
        model = self.trainer.best_model
        X     = self.trainer.X_test
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_best_model_has_predict_proba(self):
        model = self.trainer.best_model
        X     = self.trainer.X_test[:5]
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (5, 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestModelEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.train_model    import ModelTrainer
        from src.evaluate_model import ModelEvaluator
        df = _make_processed_df(200)
        trainer = ModelTrainer()
        trainer.run(df)
        cls.ev = ModelEvaluator()
        cls.ev.load_models()
        cls.ev.prepare_test_data(df)
        cls.results = cls.ev.evaluate_all()

    def test_results_non_empty(self):
        self.assertGreater(len(self.results), 0)

    def test_metrics_keys_present(self):
        for name, m in self.results.items():
            with self.subTest(model=name):
                for key in ("accuracy", "precision", "recall", "f1"):
                    self.assertIn(key, m)

    def test_confusion_matrix_figure(self):
        import matplotlib.pyplot as plt
        name = next(iter(self.ev.models))
        fig  = self.ev.plot_confusion_matrix(name)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_metrics_dataframe_shape(self):
        df = self.ev.get_metrics_dataframe()
        self.assertGreater(len(df), 0)
        self.assertIn("F1 Score", df.columns)

    def test_roc_plot(self):
        import matplotlib.pyplot as plt
        fig = self.ev.plot_roc_curves()
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
