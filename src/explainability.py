"""
explainability.py - SHAP-based Explainable AI for model predictions.
Provides feature importance, waterfall plots, and summary plots.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import BEST_MODEL_PATH, ALL_MODEL_FEATURES
from src.utils import setup_logger, load_pickle

logger = setup_logger("Explainability")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability features disabled.")


class ExplainabilityEngine:
    """
    Wraps SHAP to explain predictions made by the best model.

    Usage:
        engine   = ExplainabilityEngine()
        engine.load_model()
        fig_wf   = engine.waterfall_plot(X_instance, feature_names)
        fig_sum  = engine.summary_plot(X_background)
        shap_df  = engine.feature_importance_df(X_background)
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path) if model_path else BEST_MODEL_PATH
        self._pipeline  = None
        self._explainer = None
        self._feature_names = ALL_MODEL_FEATURES

    # ── Lazy loading ─────────────────────────────────────────────────────

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = load_pickle(self.model_path)
        return self._pipeline

    def load_model(self) -> None:
        _ = self.pipeline    # trigger load
        logger.info("Model pipeline loaded for SHAP.")

    # ── SHAP Explainer Creation ──────────────────────────────────────────

    def _get_explainer(self, X_background: np.ndarray):
        """Create or return a cached SHAP TreeExplainer / KernelExplainer."""
        if self._explainer is not None:
            return self._explainer

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Run: pip install shap")

        clf = self.pipeline.named_steps["clf"]
        scaler = self.pipeline.named_steps.get("scaler")

        if scaler is not None:
            X_bg_scaled = scaler.transform(X_background)
        else:
            X_bg_scaled = X_background

        try:
            # TreeExplainer works for RF / XGBoost
            self._explainer = shap.TreeExplainer(clf)
            logger.info("Using SHAP TreeExplainer.")
        except Exception:
            # Fallback for linear models
            self._explainer = shap.KernelExplainer(
                clf.predict_proba,
                shap.sample(X_bg_scaled, min(100, len(X_bg_scaled))),
            )
            logger.info("Using SHAP KernelExplainer.")

        return self._explainer

    def _scale(self, X: np.ndarray) -> np.ndarray:
        scaler = self.pipeline.named_steps.get("scaler")
        return scaler.transform(X) if scaler else X

    # ── Public Plots ─────────────────────────────────────────────────────

    def waterfall_plot(self,
                       X_instance: np.ndarray,
                       feature_names: list | None = None) -> plt.Figure:
        """
        SHAP waterfall plot for a single prediction.

        Args:
            X_instance:    2D array of shape (1, n_features).
            feature_names: Optional list of feature names.

        Returns:
            Matplotlib Figure.
        """
        if not SHAP_AVAILABLE:
            return self._unavailable_fig("SHAP not installed")

        names    = feature_names or self._feature_names
        explainer = self._get_explainer(X_instance)
        X_scaled  = self._scale(X_instance)

        shap_values = explainer(X_scaled)

        # Handle multi-class output (take class 1 = Pass)
        sv = shap_values
        if hasattr(sv, "values") and sv.values.ndim == 3:
            sv = sv[:, :, 1]

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.sca(ax)
        shap.waterfall_plot(sv[0], max_display=15, show=False)
        fig = plt.gcf()
        fig.suptitle("SHAP Explanation — Individual Prediction", y=1.01, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def summary_plot(self,
                     X_background: np.ndarray,
                     feature_names: list | None = None,
                     plot_type: str = "bar") -> plt.Figure:
        """
        SHAP summary / bar plot over a background dataset.

        Args:
            X_background:  2D array of shape (n_samples, n_features).
            feature_names: Optional list of feature names.
            plot_type:     "bar" or "dot".

        Returns:
            Matplotlib Figure.
        """
        if not SHAP_AVAILABLE:
            return self._unavailable_fig("SHAP not installed")

        names    = feature_names or self._feature_names
        explainer = self._get_explainer(X_background)
        X_scaled  = self._scale(X_background)

        shap_values = explainer.shap_values(X_scaled)

        # For binary classifiers → take class 1 values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(9, 6))
        plt.sca(ax)
        shap.summary_plot(
            shap_values, X_scaled,
            feature_names=names,
            plot_type=plot_type,
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle("SHAP Feature Importance Summary", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def feature_importance_df(self,
                               X_background: np.ndarray,
                               feature_names: list | None = None) -> pd.DataFrame:
        """
        Compute mean absolute SHAP values as a ranked DataFrame.

        Returns:
            DataFrame with columns: Feature, SHAP Importance.
        """
        if not SHAP_AVAILABLE:
            return pd.DataFrame({"Feature": ["SHAP not installed"], "SHAP Importance": [0]})

        names    = feature_names or self._feature_names
        explainer = self._get_explainer(X_background)
        X_scaled  = self._scale(X_background)

        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({"Feature": names, "SHAP Importance": mean_abs})
        return df.sort_values("SHAP Importance", ascending=False).reset_index(drop=True)

    def individual_explanation(self,
                                X_instance: np.ndarray,
                                feature_names: list | None = None) -> dict:
        """
        Return a dict of feature → SHAP value for one student.
        Positive values push toward Pass; negative toward Fail.
        """
        if not SHAP_AVAILABLE:
            return {}

        names    = feature_names or self._feature_names
        explainer = self._get_explainer(X_instance)
        X_scaled  = self._scale(X_instance)

        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        else:
            vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        return {name: round(float(val), 4) for name, val in zip(names, vals)}

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _unavailable_fig(reason: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"⚠️  {reason}", ha="center", va="center",
                fontsize=13, color="red", transform=ax.transAxes)
        ax.axis("off")
        return fig


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from config.config import PROCESSED_DATA_PATH, ALL_MODEL_FEATURES

    df  = pd.read_csv(PROCESSED_DATA_PATH)
    available = [f for f in ALL_MODEL_FEATURES if f in df.columns]
    X   = df[available].dropna().values

    engine = ExplainabilityEngine()
    engine.load_model()

    importance_df = engine.feature_importance_df(X[:200], available)
    print("\n🔍  SHAP Feature Importances:")
    print(importance_df.head(10).to_string(index=False))
