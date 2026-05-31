"""
student_dashboard.py - Individual student analysis, prediction, and recommendations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_DATA_PATH, BEST_MODEL_PATH, ALL_MODEL_FEATURES,
    RISK_COLORS, DEPARTMENTS, GENDERS,
)
from src.predict               import StudentPredictor
from src.recommendation_engine import RecommendationEngine
from src.explainability        import ExplainabilityEngine
from src.pdf_generator         import PDFReportGenerator
from src.utils                 import load_csv, risk_emoji, result_emoji, risk_color


# ── Cached singletons ─────────────────────────────────────────────────────

@st.cache_resource
def _get_predictor():
    return StudentPredictor()


@st.cache_resource
def _get_rec_engine():
    return RecommendationEngine()


@st.cache_resource
def _get_explainer():
    return ExplainabilityEngine()


@st.cache_resource
def _get_pdf_gen():
    return PDFReportGenerator()


# ── Helpers ───────────────────────────────────────────────────────────────

def _radar_chart(record: dict) -> go.Figure:
    """Build a radar / spider chart of the student's academic metrics."""
    categories = [
        "Attendance", "Assignment", "Quiz", "Lab", "Midterm",
        "LMS Activity", "Engagement",
    ]
    values = [
        record.get("attendance_percentage",    0),
        record.get("assignment_score",         0),
        record.get("quiz_score",               0),
        record.get("lab_score",                0),
        record.get("midterm_marks",            0),
        record.get("lms_activity",             0),
        min(record.get("videos_watched", 0) / 60 * 100, 100),   # normalise to 100
    ]
    values += [values[0]]           # close the polygon
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(30,58,95,0.25)",
        line=dict(color="#1E3A5F", width=2),
        name="Student",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[75] * len(categories), theta=categories,
        line=dict(color="red", dash="dash", width=1.5),
        name="Target (75)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickfont_size=9)),
        showlegend=True,
        margin=dict(l=30, r=30, t=40, b=30),
        height=350,
    )
    return fig


def _gauge_chart(risk_score: float, risk_level: str) -> go.Figure:
    color = risk_color(risk_level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score (0 = Safe, 100 = High Risk)"},
        gauge={
            "axis":     {"range": [0, 100]},
            "bar":      {"color": color, "thickness": 0.3},
            "steps":    [
                {"range": [0,  40],  "color": "#D5E8D4"},
                {"range": [40, 70],  "color": "#FFE6CC"},
                {"range": [70, 100], "color": "#F8CECC"},
            ],
            "threshold": {"value": risk_score, "line": {"color": color, "width": 4}},
        },
        number={"suffix": "%", "font": {"size": 40}},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ── Main Render ───────────────────────────────────────────────────────────

def render_student_dashboard():
    """Entry point called by dashboard.py."""
    st.markdown("## 🎓 Student Analysis & Prediction")

    # ── Input Method ─────────────────────────────────────────────────────
    input_method = st.radio(
        "How would you like to enter student data?",
        ["📝 Manual Entry", "📂 Select from Dataset"],
        horizontal=True,
    )

    record = None

    if input_method == "📝 Manual Entry":
        record = _manual_entry_form()
    else:
        record = _select_from_dataset()

    if record is None:
        return

    st.divider()

    # ── Run Prediction ────────────────────────────────────────────────────
    predictor  = _get_predictor()
    rec_engine = _get_rec_engine()

    with st.spinner("🔮 Running prediction…"):
        try:
            prediction = predictor.predict_single(record)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("Please train the models first using the sidebar.")
            return

    recommendations = rec_engine.generate(record, prediction["risk_level"])

    # ── Result Banner ─────────────────────────────────────────────────────
    rl = prediction["risk_level"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            f"{result_emoji(prediction['predicted_result'])} Predicted Result",
            prediction["predicted_result"],
        )
    with col_b:
        st.metric(
            f"{risk_emoji(rl)} Risk Level",
            rl,
        )
    with col_c:
        st.metric("🎯 Confidence", f"{prediction['confidence']:.1f}%")

    st.divider()

    # ── Gauge + Radar ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Risk Score Gauge")
        st.plotly_chart(_gauge_chart(prediction["risk_score"], rl), use_container_width=True)
    with col2:
        st.subheader("🕸️ Academic Performance Radar")
        st.plotly_chart(_radar_chart(record), use_container_width=True)

    st.divider()

    # ── Recommendations ───────────────────────────────────────────────────
    st.subheader("💡 Personalised Recommendations")
    priority_colors = {
        "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
    }
    for r in recommendations:
        badge = priority_colors.get(r["priority"], "⚪")
        with st.expander(f"{r['icon']} {badge} [{r['priority']}] {r['category']}"):
            st.write(r["message"])
            st.info(f"✅ **Action:** {r['action']}")

    st.divider()

    # ── SHAP Explanation ──────────────────────────────────────────────────
    with st.expander("🔍 Explainable AI — Why this prediction?"):
        try:
            explainer = _get_explainer()
            explainer.load_model()

            # Build feature array for this student
            df_tmp = pd.DataFrame([record])
            from src.data_preprocessing  import DataPreprocessor
            from src.feature_engineering import FeatureEngineer
            pp = DataPreprocessor()
            fe = FeatureEngineer()
            df_tmp = pp.preprocess_single(record)
            df_tmp = fe.run_single(df_tmp)
            available = [f for f in ALL_MODEL_FEATURES if f in df_tmp.columns]
            X_inst = df_tmp[available].values

            shap_dict = explainer.individual_explanation(X_inst, available)
            if shap_dict:
                shap_df = pd.DataFrame(
                    list(shap_dict.items()), columns=["Feature", "SHAP Value"]
                ).sort_values("SHAP Value", ascending=False)

                fig_shap = px.bar(
                    shap_df, x="SHAP Value", y="Feature", orientation="h",
                    color="SHAP Value",
                    color_continuous_scale=["#E74C3C", "#ECF0F1", "#27AE60"],
                    template="plotly_white",
                    title="SHAP Feature Contributions (green = toward Pass, red = toward Fail)",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("SHAP explanations not available (install `pip install shap`).")
        except Exception as exc:
            st.warning(f"SHAP explanation unavailable: {exc}")

    st.divider()

    # ── PDF Report ────────────────────────────────────────────────────────
    st.subheader("📄 Download Report")
    if st.button("📥 Generate PDF Report"):
        try:
            gen       = _get_pdf_gen()
            pdf_bytes = gen.generate(record, prediction, recommendations)
            sid       = record.get("student_id", "student")
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"report_{sid}.pdf",
                mime="application/pdf",
            )
            st.success("✅ Report generated successfully!")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}. Install `pip install reportlab`.")


# ── Form Helpers ──────────────────────────────────────────────────────────

def _manual_entry_form() -> dict | None:
    with st.form("student_form"):
        st.markdown("### 🧑‍🎓 Student Information")
        c1, c2, c3 = st.columns(3)
        sid  = c1.text_input("Student ID",  value="STU0001")
        name = c2.text_input("Name",        value="Arjun Kumar")
        gender = c3.selectbox("Gender",     GENDERS)

        c4, c5 = st.columns(2)
        age   = c4.number_input("Age", 15, 35, 20)
        dept  = c5.selectbox("Department",  DEPARTMENTS)

        st.markdown("### 📚 Academic Scores")
        a1, a2, a3 = st.columns(3)
        att  = a1.slider("Attendance (%)",     0.0, 100.0, 75.0, 0.5)
        asn  = a2.slider("Assignment Score",   0.0, 100.0, 65.0, 0.5)
        quiz = a3.slider("Quiz Score",         0.0, 100.0, 60.0, 0.5)
        b1, b2 = st.columns(2)
        lab  = b1.slider("Lab Score",          0.0, 100.0, 70.0, 0.5)
        mid  = b2.slider("Midterm Marks",      0.0, 100.0, 55.0, 0.5)

        st.markdown("### 💻 Engagement")
        e1, e2, e3 = st.columns(3)
        lms  = e1.slider("LMS Activity",       0.0, 100.0, 50.0, 0.5)
        vid  = e2.number_input("Videos Watched",       0, 60, 15)
        disc = e3.number_input("Discussion Participation", 0, 50, 8)

        submitted = st.form_submit_button("🚀 Predict", type="primary")

    if submitted:
        return {
            "student_id":               sid,
            "name":                     name,
            "gender":                   gender,
            "age":                      age,
            "department":               dept,
            "attendance_percentage":    att,
            "assignment_score":         asn,
            "quiz_score":               quiz,
            "lab_score":                lab,
            "midterm_marks":            mid,
            "lms_activity":             lms,
            "videos_watched":           vid,
            "discussion_participation": disc,
        }
    return None


def _select_from_dataset() -> dict | None:
    try:
        df = load_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        st.warning("Dataset not found. Please run the preprocessing pipeline first.")
        return None

    if "student_id" not in df.columns:
        st.error("student_id column missing from processed data.")
        return None

    # Decode label-encoded columns back to strings for display
    ids = df["student_id"].tolist()
    chosen_id = st.selectbox("Select Student ID", ids)
    row = df[df["student_id"] == chosen_id].iloc[0].to_dict()
    st.info(f"Selected: **{row.get('name', chosen_id)}** — {row.get('department', '')}")
    return row
