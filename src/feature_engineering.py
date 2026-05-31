"""
feature_engineering.py - Creates domain-driven features for the ML models.

New Features Created:
  • attendance_risk              – binary/ordinal risk from attendance
  • assignment_completion_ratio  – assignment score normalised to 0-1
  • engagement_score             – composite LMS engagement metric
  • performance_trend            – improvement trend between early & late assessments
  • overall_academic_score       – weighted composite of all academic scores
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_DATA_PATH, PROCESSED_DIR,
    NUMERIC_FEATURES, ENGINEERED_FEATURES,
)
from src.utils import setup_logger, save_csv, load_csv

logger = setup_logger("FeatureEngineering")


class FeatureEngineer:
    """
    Adds derived / engineered features to the student DataFrame.
    Call `run()` to transform the processed DataFrame in-place.
    """

    # Attendance thresholds (%)
    ATTENDANCE_HIGH   = 85
    ATTENDANCE_MEDIUM = 75

    # Weights for the overall academic score
    SCORE_WEIGHTS = {
        "attendance_percentage": 0.20,
        "assignment_score":      0.20,
        "midterm_marks":         0.30,
        "quiz_score":            0.15,
        "lab_score":             0.15,
    }

    def __init__(self):
        self.feature_stats: dict = {}

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame | None = None,
            save: bool = True) -> pd.DataFrame:
        """
        Full feature-engineering pipeline.

        Args:
            df:   DataFrame to transform; loads PROCESSED_DATA_PATH if None.
            save: Whether to overwrite PROCESSED_DATA_PATH with the result.

        Returns:
            DataFrame with all engineered columns appended.
        """
        logger.info("=== Feature Engineering START ===")
        if df is None:
            df = load_csv(PROCESSED_DATA_PATH)

        df = self._attendance_risk(df)
        df = self._assignment_completion_ratio(df)
        df = self._engagement_score(df)
        df = self._performance_trend(df)
        df = self._overall_academic_score(df)

        self._compute_stats(df)

        if save:
            save_csv(df, PROCESSED_DATA_PATH)

        logger.info(f"=== Feature Engineering DONE — added {len(ENGINEERED_FEATURES)} features ===")
        return df

    def run_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply engineering to a single-row inference DataFrame."""
        df = self._attendance_risk(df)
        df = self._assignment_completion_ratio(df)
        df = self._engagement_score(df)
        df = self._performance_trend(df)
        df = self._overall_academic_score(df)
        return df

    # ── Feature Creators ────────────────────────────────────────────────

    def _attendance_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        0 = high risk (attendance < 75%)
        1 = medium risk (75 ≤ attendance < 85%)
        2 = low risk (attendance ≥ 85%)
        """
        att = df["attendance_percentage"]
        df["attendance_risk"] = np.where(
            att >= self.ATTENDANCE_HIGH, 2,
            np.where(att >= self.ATTENDANCE_MEDIUM, 1, 0)
        )
        logger.debug("attendance_risk created.")
        return df

    @staticmethod
    def _assignment_completion_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """Normalised assignment score: assignment_score / 100."""
        df["assignment_completion_ratio"] = (
            df["assignment_score"].clip(0, 100) / 100
        ).round(4)
        logger.debug("assignment_completion_ratio created.")
        return df

    @staticmethod
    def _engagement_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite engagement on a 0-100 scale.
        Weighted average of:
          lms_activity (40%), videos_watched normalised (30%),
          discussion_participation normalised (30%)
        """
        # Normalise videos (max 60) and discussions (max 50) to 0-100
        vid_norm  = (df["videos_watched"].clip(0, 60)          / 60  * 100)
        disc_norm = (df["discussion_participation"].clip(0, 50) / 50  * 100)
        lms       = df["lms_activity"].clip(0, 100)

        df["engagement_score"] = (
            0.40 * lms + 0.30 * vid_norm + 0.30 * disc_norm
        ).round(2)
        logger.debug("engagement_score created.")
        return df

    @staticmethod
    def _performance_trend(df: pd.DataFrame) -> pd.DataFrame:
        """
        Difference between later-stage scores (midterm) and earlier scores
        (assignment + quiz average).  Positive → improving student.
        Range is roughly –100 to +100.
        """
        early = (df["assignment_score"] + df["quiz_score"]) / 2
        late  = df["midterm_marks"]
        df["performance_trend"] = (late - early).round(2)
        logger.debug("performance_trend created.")
        return df

    def _overall_academic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted composite of attendance + assessments.
        Normalised to 0-100.
        """
        score = sum(
            df[col] * weight
            for col, weight in self.SCORE_WEIGHTS.items()
            if col in df.columns
        )
        df["overall_academic_score"] = score.clip(0, 100).round(2)
        logger.debug("overall_academic_score created.")
        return df

    # ── Stats ────────────────────────────────────────────────────────────

    def _compute_stats(self, df: pd.DataFrame) -> None:
        """Compute and log basic stats for engineered features."""
        self.feature_stats = {}
        for feat in ENGINEERED_FEATURES:
            if feat in df.columns:
                s = df[feat].describe()
                self.feature_stats[feat] = s.to_dict()
                logger.info(
                    f"  {feat}: mean={s['mean']:.2f}, std={s['std']:.2f}, "
                    f"min={s['min']:.2f}, max={s['max']:.2f}"
                )


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    fe = FeatureEngineer()
    df_out = fe.run()
    print(f"\n✅  Feature-engineered shape: {df_out.shape}")
    print(df_out[ENGINEERED_FEATURES].head(5).to_string())
