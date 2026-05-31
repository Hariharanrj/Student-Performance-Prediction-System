"""
admin_dashboard.py - Admin overview dashboard with analytics and model metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express     as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_DATA_PATH, BEST_MODEL_PATH, RF_MODEL_PATH, LR_MODEL_PATH, XGB_MODEL_PATH,
    ALL_MODEL_FEATURES, TARGET_COLUMN, RISK_COLORS,
)
from src.predict         import StudentPredictor
from src.evaluate_model  import ModelEvaluator
from src.utils           import load_csv, load_pickle, check_model_files


# ── Helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_processed_data() -> pd.DataFrame:
    try:
        return load_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _run_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    predictor = StudentPredictor()
    return predictor.predict_batch(df)


@st.cache_resource
def _get_evaluator() -> ModelEvaluator:
    ev = ModelEvaluator()
    ev.load_models()
    ev.prepare_test_data()
    return ev


# ── Main Render ───────────────────────────────────────────────────────────

def render_admin_dashboard():
    """Entry point called by dashboard.py."""
    st.markdown("## 🏫 Admin Dashboard — System Overview")

    # ── Load data ────────────────────────────────────────────────────────
    df_raw = _load_processed_data()
    if df_raw.empty:
        st.warning("⚠️  No processed data found. Please run the training pipeline first.")
        return

    df = _run_batch_predictions(df_raw)

    # ── KPI Cards ────────────────────────────────────────────────────────
    total     = len(df)
    high_risk = (df.get("risk_level", pd.Series()) == "High Risk").sum()
    avg_att   = df.get("attendance_percentage", pd.Series(dtype=float)).mean()
    pass_rate = (df.get("predicted_result", pd.Series()) == "Pass").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Total Students",   total)
    with c2:
        st.metric("🔴 High Risk",        int(high_risk),
                  delta=f"{high_risk/total*100:.1f}%" if total else None,
                  delta_color="inverse")
    with c3:
        st.metric("📅 Avg Attendance",   f"{avg_att:.1f}%")
    with c4:
        st.metric("✅ Predicted Pass Rate", f"{pass_rate:.1f}%")

    st.divider()

    # ── Row 1: Attendance + Risk Distribution ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Attendance Distribution")
        fig_att = px.histogram(
            df, x="attendance_percentage",
            nbins=20, color_discrete_sequence=["#1E3A5F"],
            labels={"attendance_percentage": "Attendance (%)"},
            template="plotly_white",
        )
        fig_att.add_vline(x=75, line_dash="dash", line_color="red",
                          annotation_text="75% Threshold")
        st.plotly_chart(fig_att, use_container_width=True)

    with col2:
        st.subheader("🎯 Risk Level Distribution")
        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_risk = px.pie(
                risk_counts, names="Risk Level", values="Count",
                color="Risk Level", color_discrete_map=RISK_COLORS,
                template="plotly_white",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

    # ── Row 2: Performance Scatter + Pass/Fail Bar ────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📈 Attendance vs Midterm Marks")
        if "midterm_marks" in df.columns and "risk_level" in df.columns:
            fig_sc = px.scatter(
                df, x="attendance_percentage", y="midterm_marks",
                color="risk_level", color_discrete_map=RISK_COLORS,
                opacity=0.65, template="plotly_white",
                labels={
                    "attendance_percentage": "Attendance (%)",
                    "midterm_marks": "Midterm Marks",
                    "risk_level": "Risk Level",
                },
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    with col4:
        st.subheader("🏆 Department-wise Performance")
        if "department" in df.columns and "predicted_result" in df.columns:
            dept_perf = (
                df.groupby("department")["predicted_result"]
                .apply(lambda s: (s == "Pass").mean() * 100)
                .reset_index()
            )
            dept_perf.columns = ["Department", "Pass Rate (%)"]
            dept_perf["Short"] = dept_perf["Department"].astype(str).str.split().str[:2].str.join(" ")
            fig_dept = px.bar(
                dept_perf, x="Short", y="Pass Rate (%)",
                color="Pass Rate (%)", color_continuous_scale="Blues",
                template="plotly_white",
            )
            st.plotly_chart(fig_dept, use_container_width=True)

    st.divider()

    # ── Model Comparison ─────────────────────────────────────────────────
    st.subheader("🤖 Model Evaluation Metrics")

    if check_model_files([RF_MODEL_PATH, LR_MODEL_PATH]):
        with st.spinner("Evaluating models…"):
            try:
                ev      = _get_evaluator()
                results = ev.evaluate_all()
                metrics_df = ev.get_metrics_dataframe()

                st.dataframe(
                    metrics_df.style.highlight_max(axis=0, color="#D5E8D4")
                               .format("{:.4f}"),
                    use_container_width=True,
                )

                col5, col6 = st.columns(2)
                with col5:
                    st.pyplot(ev.plot_metrics_bar(), use_container_width=True)
                with col6:
                    st.pyplot(ev.plot_roc_curves(), use_container_width=True)

                st.subheader("🔀 Confusion Matrix")
                model_choice = st.selectbox("Select model", list(ev.models.keys()))
                st.pyplot(ev.plot_confusion_matrix(model_choice), use_container_width=True)

            except Exception as exc:
                st.error(f"Could not evaluate models: {exc}")
    else:
        st.info("ℹ️  Train models first using the sidebar or `main.py`.")

    st.divider()

    # ── High-Risk Student Table ───────────────────────────────────────────
    st.subheader("🚨 High-Risk Students")
    if "risk_level" in df.columns:
        high_df = df[df["risk_level"] == "High Risk"].copy()
        if high_df.empty:
            st.success("✅  No high-risk students detected!")
        else:
            display_cols = [c for c in [
                "student_id", "name", "department",
                "attendance_percentage", "midterm_marks",
                "risk_score", "predicted_result",
            ] if c in high_df.columns]
            st.dataframe(
                high_df[display_cols]
                .sort_values("risk_score", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
            )
