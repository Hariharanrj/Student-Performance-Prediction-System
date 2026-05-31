"""
train_model.py - Trains Logistic Regression, Random Forest, and XGBoost;
                 selects the best model by F1-score and saves all artefacts.
"""

import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline        import Pipeline

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_DATA_PATH, MODEL_CONFIG,
    BEST_MODEL_PATH, RF_MODEL_PATH, LR_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH,
    ALL_MODEL_FEATURES, TARGET_COLUMN,
    TRAINED_DIR, SCALER_DIR,
)
from src.utils import setup_logger, save_pickle

logger = setup_logger("ModelTraining")


class ModelTrainer:
    """
    Trains and persists three classifiers, chooses the best by CV F1-score.

    Attributes:
        results:    dict mapping model name → evaluation metrics dict.
        best_name:  name of the best model.
        best_model: the best fitted Pipeline.
        X_test, y_test: held-out test split (used by ModelEvaluator).
    """

    def __init__(self):
        self.results:    dict = {}
        self.best_name:  str  = ""
        self.best_model       = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.scaler          = None

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame | None = None) -> dict:
        """
        Full training pipeline.

        Args:
            df: Pre-processed + feature-engineered DataFrame.
                Loads PROCESSED_DATA_PATH if None.

        Returns:
            dict of model results with accuracy, precision, recall, F1.
        """
        logger.info("=== Model Training START ===")

        if df is None:
            df = pd.read_csv(PROCESSED_DATA_PATH)

        X, y = self._prepare_features(df)
        self._split(X, y)
        self._train_all()
        self._save_artefacts()

        logger.info(f"=== Model Training DONE — best model: {self.best_name} ===")
        return self.results

    # ── Private Helpers ──────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame):
        """Select model features; drop rows with NaN in feature cols."""
        available = [f for f in ALL_MODEL_FEATURES if f in df.columns]
        missing   = [f for f in ALL_MODEL_FEATURES if f not in df.columns]
        if missing:
            logger.warning(f"Features not found (skipped): {missing}")

        df_clean = df[available + [TARGET_COLUMN]].dropna()
        X = df_clean[available].values
        y = df_clean[TARGET_COLUMN].values

        logger.info(f"  Feature matrix: {X.shape},  class balance: {np.bincount(y.astype(int))}")
        self._feature_names = available
        return X, y

    def _split(self, X, y) -> None:
        cfg = MODEL_CONFIG
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            stratify=y,
        )
        logger.info(f"  Train: {len(self.X_train)}  |  Test: {len(self.X_test)}")

    def _build_pipelines(self) -> dict:
        cfg = MODEL_CONFIG
        self.scaler = StandardScaler()

        pipelines = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**cfg["logistic_regression"])),
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(**cfg["random_forest"])),
            ]),
        }

        if XGB_AVAILABLE:
            xgb_params = {k: v for k, v in cfg["xgboost"].items()
                          if k != "use_label_encoder"}
            pipelines["XGBoost"] = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(**xgb_params, verbosity=0)),
            ])
        else:
            logger.warning("XGBoost not installed — skipping XGBoost model.")

        return pipelines

    def _train_all(self) -> None:
        """Train each pipeline, evaluate with cross-validation + test set."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )

        cv = StratifiedKFold(
            n_splits=MODEL_CONFIG["cv_folds"],
            shuffle=True,
            random_state=MODEL_CONFIG["random_state"],
        )

        best_f1   = -1.0
        pipelines = self._build_pipelines()

        for name, pipe in pipelines.items():
            logger.info(f"  Training {name} …")
            pipe.fit(self.X_train, self.y_train)

            # Cross-validation on training split
            cv_scores = cross_val_score(
                pipe, self.X_train, self.y_train,
                cv=cv, scoring="f1", n_jobs=-1,
            )

            # Test-set metrics
            y_pred = pipe.predict(self.X_test)
            metrics = {
                "accuracy":  round(accuracy_score (self.y_test, y_pred), 4),
                "precision": round(precision_score(self.y_test, y_pred, zero_division=0), 4),
                "recall":    round(recall_score   (self.y_test, y_pred, zero_division=0), 4),
                "f1":        round(f1_score        (self.y_test, y_pred, zero_division=0), 4),
                "cv_f1_mean": round(cv_scores.mean(), 4),
                "cv_f1_std":  round(cv_scores.std(),  4),
            }
            self.results[name] = metrics
            logger.info(
                f"    {name}: Acc={metrics['accuracy']:.4f}  "
                f"F1={metrics['f1']:.4f}  CV-F1={metrics['cv_f1_mean']:.4f}"
            )

            # Track best by cross-validated F1
            if metrics["cv_f1_mean"] > best_f1:
                best_f1         = metrics["cv_f1_mean"]
                self.best_name  = name
                self.best_model = pipe

            # Save individual models
            _paths = {
                "Random Forest":        RF_MODEL_PATH,
                "Logistic Regression":  LR_MODEL_PATH,
                "XGBoost":              XGB_MODEL_PATH,
            }
            if name in _paths:
                save_pickle(pipe, _paths[name])

        logger.info(f"  Best model: {self.best_name} (CV-F1={best_f1:.4f})")

    def _save_artefacts(self) -> None:
        """Save best model and scaler."""
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        SCALER_DIR.mkdir(parents=True, exist_ok=True)

        save_pickle(self.best_model, BEST_MODEL_PATH)

        # Also save a standalone scaler for SHAP explainer
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        save_pickle(scaler, SCALER_PATH)

        logger.info(f"  Artefacts saved → {TRAINED_DIR}")


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run()
    print("\n📊  Training Results:")
    for model, metrics in results.items():
        print(f"  {model:25s}  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  CV-F1={metrics['cv_f1_mean']:.4f}")
    print(f"\n🏆  Best Model: {trainer.best_name}")
