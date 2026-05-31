"""
predict.py - Handles single-student and batch predictions with risk scoring.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BEST_MODEL_PATH, ALL_MODEL_FEATURES, RISK_THRESHOLDS, RISK_LABELS,
    TARGET_COLUMN,
)
from src.data_preprocessing  import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.utils import setup_logger, load_pickle

logger = setup_logger("Prediction")


class StudentPredictor:
    """
    Loads the best saved model and provides:
      • predict_single(record)  – predicts one student at a time
      • predict_batch(df)       – predicts a whole DataFrame
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path) if model_path else BEST_MODEL_PATH
        self._model     = None
        self._preprocessor = DataPreprocessor()
        self._feature_engineer = FeatureEngineer()

    # ── Lazy model loading ───────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            self._model = load_pickle(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        return self._model

    # ── Public API ──────────────────────────────────────────────────────

    def predict_single(self, record: dict) -> dict:
        """
        Predict Pass/Fail and risk level for one student.

        Args:
            record: dict with raw feature values (may include student_id, name).

        Returns:
            dict with keys: predicted_result, fail_probability, risk_score,
                             risk_level, confidence, pass_probability.
        """
        # Preserve metadata
        student_id = record.get("student_id", "UNKNOWN")
        name       = record.get("name",       "Unknown")

        df = pd.DataFrame([record])
        df = self._preprocess_and_engineer(df)

        X  = self._select_features(df)
        return self._make_prediction(X, student_id, name)

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for all rows in a DataFrame.

        Args:
            df_raw: DataFrame of raw student records (may or may not have 'result').

        Returns:
            Original DataFrame augmented with prediction columns.
        """
        df_work = df_raw.copy()

        # Drop ground-truth if present (inference only)
        if TARGET_COLUMN in df_work.columns:
            df_work = df_work.drop(columns=[TARGET_COLUMN])

        df_processed = self._preprocess_and_engineer(df_work)
        X = self._select_features(df_processed)

        proba        = self.model.predict_proba(X)
        fail_prob    = proba[:, 0]          # probability of class 0 = Fail
        pass_prob    = proba[:, 1]
        predictions  = self.model.predict(X)

        df_raw = df_raw.copy()
        df_raw["predicted_result"]  = ["Pass" if p == 1 else "Fail" for p in predictions]
        df_raw["pass_probability"]  = np.round(pass_prob,  4)
        df_raw["fail_probability"]  = np.round(fail_prob,  4)
        df_raw["risk_score"]        = np.round(fail_prob * 100, 2)   # 0-100
        df_raw["risk_level"]        = [self._classify_risk(fp) for fp in fail_prob]
        df_raw["confidence"]        = np.round(np.maximum(pass_prob, fail_prob) * 100, 2)

        logger.info(f"Batch prediction complete: {len(df_raw)} rows.")
        return df_raw

    # ── Private Helpers ──────────────────────────────────────────────────

    def _preprocess_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run preprocessing + feature engineering on a copy of df."""
        df = self._preprocessor.preprocess_single(df.to_dict(orient="records")[0]) \
            if len(df) == 1 else self._run_batch_preprocessing(df)
        df = self._feature_engineer.run_single(df)
        return df

    def _run_batch_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lightweight batch-safe preprocessing (no target encoding needed)."""
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Fill numeric NaN with median / 0
        for col in ALL_MODEL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Encode gender / department
        from sklearn.preprocessing import LabelEncoder
        for col in ["gender", "department"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def _select_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract model features in the correct order."""
        available = [f for f in ALL_MODEL_FEATURES if f in df.columns]
        missing   = set(ALL_MODEL_FEATURES) - set(available)
        if missing:
            for col in missing:
                df[col] = 0.0
        return df[ALL_MODEL_FEATURES].values

    def _make_prediction(self, X: np.ndarray,
                         student_id: str, name: str) -> dict:
        proba      = self.model.predict_proba(X)[0]
        pred_class = self.model.predict(X)[0]

        fail_prob  = float(proba[0])
        pass_prob  = float(proba[1])
        risk_level = self._classify_risk(fail_prob)

        result = {
            "student_id":       student_id,
            "name":             name,
            "predicted_result": "Pass" if pred_class == 1 else "Fail",
            "pass_probability": round(pass_prob,  4),
            "fail_probability": round(fail_prob,  4),
            "risk_score":       round(fail_prob * 100, 2),
            "risk_level":       risk_level,
            "confidence":       round(max(pass_prob, fail_prob) * 100, 2),
        }

        logger.info(
            f"Prediction for {student_id} ({name}): "
            f"{result['predicted_result']} | Risk={risk_level} | "
            f"Score={result['risk_score']:.1f}"
        )
        return result

    @staticmethod
    def _classify_risk(fail_prob: float) -> str:
        """Convert a failure probability to a risk label."""
        if fail_prob >= RISK_THRESHOLDS["high"]:
            return RISK_LABELS["high"]
        if fail_prob >= RISK_THRESHOLDS["medium"]:
            return RISK_LABELS["medium"]
        return RISK_LABELS["low"]


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from config.config import SAMPLE_DIR

    predictor = StudentPredictor()

    # Single prediction demo
    sample_record = {
        "student_id":               "STU9999",
        "name":                     "Demo Student",
        "gender":                   "Male",
        "age":                      20,
        "department":               "Computer Science Engineering",
        "attendance_percentage":    55.0,
        "assignment_score":         45.0,
        "quiz_score":               40.0,
        "lab_score":                50.0,
        "midterm_marks":            38.0,
        "lms_activity":             30.0,
        "videos_watched":           10,
        "discussion_participation": 5,
    }

    result = predictor.predict_single(sample_record)
    print("\n🔮  Single Prediction:")
    for k, v in result.items():
        print(f"   {k:22s}: {v}")
