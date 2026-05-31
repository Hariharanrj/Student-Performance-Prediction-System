"""
data_preprocessing.py - Cleans and prepares raw student data for ML training.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN,
    ID_COLUMN, NAME_COLUMN,
)
from src.utils import setup_logger, save_csv, load_csv

logger = setup_logger("Preprocessing")


class DataPreprocessor:
    """
    Handles all preprocessing steps:
      - Loading raw CSV
      - Missing value imputation
      - Outlier capping
      - Label encoding of categorical columns
      - Target encoding (Pass→1, Fail→0)
      - Saving processed data
    """

    REQUIRED_COLS = [
        "student_id", "name", "gender", "age", "department",
        "attendance_percentage", "assignment_score", "quiz_score",
        "lab_score", "midterm_marks", "lms_activity",
        "videos_watched", "discussion_participation",
    ]

    NUMERIC_RANGES = {
        "age":                      (15, 35),
        "attendance_percentage":    (0, 100),
        "assignment_score":         (0, 100),
        "quiz_score":               (0, 100),
        "lab_score":                (0, 100),
        "midterm_marks":            (0, 100),
        "lms_activity":             (0, 100),
        "videos_watched":           (0, 200),
        "discussion_participation": (0, 100),
    }

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.df_raw: pd.DataFrame | None  = None
        self.df_processed: pd.DataFrame | None = None

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            df: Optional DataFrame to process; if None, loads RAW_DATA_PATH.

        Returns:
            Processed DataFrame saved to PROCESSED_DATA_PATH.
        """
        logger.info("=== Preprocessing pipeline START ===")

        self.df_raw = df if df is not None else load_csv(RAW_DATA_PATH)
        df_work = self.df_raw.copy()

        df_work = self._standardise_columns(df_work)
        df_work = self._validate_required_columns(df_work)
        df_work = self._drop_duplicates(df_work)
        df_work = self._impute_missing_values(df_work)
        df_work = self._cap_outliers(df_work)
        df_work = self._encode_categoricals(df_work)
        df_work = self._encode_target(df_work)

        self.df_processed = df_work
        save_csv(df_work, PROCESSED_DATA_PATH)

        logger.info(f"=== Preprocessing pipeline DONE — {len(df_work)} rows ===")
        return df_work

    def preprocess_single(self, record: dict) -> pd.DataFrame:
        """
        Preprocess a single student record for inference.

        Args:
            record: dict with raw feature values.

        Returns:
            Single-row DataFrame ready for the model.
        """
        df = pd.DataFrame([record])
        df = self._standardise_columns(df)
        df = self._impute_missing_values(df, single=True)
        df = self._cap_outliers(df)
        df = self._encode_categoricals(df, fit=False)
        return df

    # ── Private Steps ────────────────────────────────────────────────────

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case and strip whitespace from all column names."""
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info("Column validation passed.")
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["student_id"]).reset_index(drop=True)
        logger.info(f"Dropped {before - len(df)} duplicate student IDs.")
        return df

    def _impute_missing_values(self, df: pd.DataFrame,
                               single: bool = False) -> pd.DataFrame:
        """Fill numeric NaNs with median, categorical NaNs with mode."""
        for col in NUMERIC_FEATURES:
            if col in df.columns and df[col].isna().any():
                fill = 0.0 if single else df[col].median()
                df[col] = df[col].fillna(fill)
                logger.debug(f"  Imputed {col} → {fill:.2f}")

        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isna().any():
                fill = "Unknown" if single else df[col].mode()[0]
                df[col] = df[col].fillna(fill)
                logger.debug(f"  Imputed {col} → {fill}")

        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip numeric values to their valid ranges."""
        for col, (lo, hi) in self.NUMERIC_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        logger.info("Outlier capping applied.")
        return df

    def _encode_categoricals(self, df: pd.DataFrame,
                              fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical features."""
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen labels gracefully
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0]
                        if x in le.classes_ else -1
                    )
                else:
                    logger.warning(f"No encoder for '{col}'; encoding with -1.")
                    df[col] = -1
        logger.info("Categorical encoding complete.")
        return df

    @staticmethod
    def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Pass/Fail to 1/0."""
        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().str.capitalize()
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Pass").astype(int)
            logger.info(f"Target distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")
        return df


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df_out = preprocessor.run()
    print(f"\n✅  Processed data shape: {df_out.shape}")
    print(df_out.head(3).to_string())
