"""
evaluate_model.py - Generates comprehensive evaluation metrics and plots
                    for all trained models.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BEST_MODEL_PATH, RF_MODEL_PATH, LR_MODEL_PATH, XGB_MODEL_PATH,
    ALL_MODEL_FEATURES, TARGET_COLUMN, PROCESSED_DATA_PATH,
)
from src.utils import setup_logger, load_pickle, load_csv

logger = setup_logger("ModelEvaluation")


class ModelEvaluator:
    """
    Loads trained models and computes full evaluation suites.

    Usage:
        evaluator = ModelEvaluator()
        results   = evaluator.evaluate_all()
        fig_cm    = evaluator.plot_confusion_matrix("Random Forest")
        fig_roc   = evaluator.plot_roc_curves()
        fig_imp   = evaluator.plot_feature_importance()
    """

    MODEL_PATHS = {
        "Random Forest":       RF_MODEL_PATH,
        "Logistic Regression": LR_MODEL_PATH,
        "XGBoost":             XGB_MODEL_PATH,
    }

    def __init__(self):
        self.models:   dict = {}
        self.X_test    = None
        self.y_test    = None
        self.results:  dict = {}
        self._feature_names: list = []

    # ── Public API ──────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load all available trained model pipelines."""
        for name, path in self.MODEL_PATHS.items():
            try:
                self.models[name] = load_pickle(path)
                logger.info(f"  Loaded: {name}")
            except FileNotFoundError:
                logger.warning(f"  Model file not found: {path} — skipping.")

    def prepare_test_data(self, df: pd.DataFrame | None = None):
        """Prepare a held-out test split from processed data."""
        from sklearn.model_selection import train_test_split
        from config.config import MODEL_CONFIG

        if df is None:
            df = load_csv(PROCESSED_DATA_PATH)

        available = [f for f in ALL_MODEL_FEATURES if f in df.columns]
        self._feature_names = available

        df_clean = df[available + [TARGET_COLUMN]].dropna()
        X = df_clean[available].values
        y = df_clean[TARGET_COLUMN].values

        _, self.X_test, _, self.y_test = train_test_split(
            X, y,
            test_size=MODEL_CONFIG["test_size"],
            random_state=MODEL_CONFIG["random_state"],
            stratify=y,
        )
        logger.info(f"Test set prepared: {self.X_test.shape}")

    def evaluate_all(self) -> dict:
        """
        Compute accuracy / precision / recall / F1 for every loaded model.

        Returns:
            dict: {model_name: {metric: value, ...}}
        """
        if not self.models:
            self.load_models()
        if self.X_test is None:
            self.prepare_test_data()

        self.results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = (
                model.predict_proba(self.X_test)[:, 1]
                if hasattr(model, "predict_proba") else None
            )

            metrics = {
                "accuracy":  round(accuracy_score (self.y_test, y_pred), 4),
                "precision": round(precision_score(self.y_test, y_pred, zero_division=0), 4),
                "recall":    round(recall_score   (self.y_test, y_pred, zero_division=0), 4),
                "f1":        round(f1_score        (self.y_test, y_pred, zero_division=0), 4),
                "report":    classification_report(self.y_test, y_pred, target_names=["Fail", "Pass"]),
            }
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                metrics["auc"]  = round(auc(fpr, tpr), 4)
                metrics["_fpr"] = fpr
                metrics["_tpr"] = tpr

            self.results[name] = metrics
            logger.info(
                f"  {name}: Acc={metrics['accuracy']:.4f}  "
                f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )

        return self.results

    # ── Plots ────────────────────────────────────────────────────────────

    def plot_confusion_matrix(self, model_name: str) -> plt.Figure:
        """Return a Matplotlib Figure with the confusion matrix for one model."""
        model  = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not loaded.")

        y_pred = model.predict(self.X_test)
        cm     = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_roc_curves(self) -> plt.Figure:
        """Return a Figure with ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")

        for name, metrics in self.results.items():
            if "_fpr" in metrics:
                ax.plot(
                    metrics["_fpr"], metrics["_tpr"],
                    lw=2,
                    label=f"{name} (AUC = {metrics.get('auc', 0):.3f})",
                )

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def plot_metrics_bar(self) -> plt.Figure:
        """Return a grouped bar chart of accuracy / precision / recall / F1."""
        if not self.results:
            raise RuntimeError("Call evaluate_all() first.")

        metrics_keys = ["accuracy", "precision", "recall", "f1"]
        names        = list(self.results.keys())
        values       = {m: [self.results[n][m] for n in names] for m in metrics_keys}

        x   = np.arange(len(names))
        w   = 0.18
        fig, ax = plt.subplots(figsize=(9, 5))

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
        for i, (metric, color) in enumerate(zip(metrics_keys, colors)):
            ax.bar(x + i * w, values[metric], w, label=metric.capitalize(), color=color)

        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Model Comparison — Evaluation Metrics", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_feature_importance(self, top_n: int = 15) -> plt.Figure:
        """Return a horizontal bar chart of feature importances (Random Forest)."""
        model = self.models.get("Random Forest") or next(iter(self.models.values()), None)
        if model is None:
            raise RuntimeError("No models loaded.")

        clf = model.named_steps.get("clf")
        if not hasattr(clf, "feature_importances_"):
            logger.warning("Model has no feature_importances_; skipping plot.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Feature importances not available", ha="center")
            return fig

        importances = clf.feature_importances_
        feat_names  = self._feature_names or [f"f{i}" for i in range(len(importances))]
        idx         = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(
            [feat_names[i] for i in idx],
            importances[idx],
            color="#1E3A5F",
        )
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(f"Top-{top_n} Feature Importances (Random Forest)",
                     fontsize=13, fontweight="bold")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        fig.tight_layout()
        return fig

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame of evaluation metrics for display."""
        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model":     name,
                "Accuracy":  m["accuracy"],
                "Precision": m["precision"],
                "Recall":    m["recall"],
                "F1 Score":  m["f1"],
                "AUC":       m.get("auc", "N/A"),
            })
        return pd.DataFrame(rows).set_index("Model")


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    ev = ModelEvaluator()
    ev.load_models()
    ev.prepare_test_data()
    results = ev.evaluate_all()

    print("\n📊  Evaluation Results:")
    for model, m in results.items():
        print(f"  {model:25s}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
