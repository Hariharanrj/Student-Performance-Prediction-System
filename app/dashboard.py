"""
dashboard.py - Main Streamlit application router.
Run with:  streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import UI_CONFIG, BEST_MODEL_PATH, PROCESSED_DATA_PATH
from src.utils import check_model_files

# ── Page Config (must be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded",
)

from app.login             import render_login, logout
from app.admin_dashboard   import render_admin_dashboard
from app.student_dashboard import render_student_dashboard


# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] { background: #1E3A5F; }
  [data-testid="stSidebar"] * { color: #FFFFFF !important; }
  [data-testid="stSidebar"] .stButton button {
      background: #2E5E9E; border: none; border-radius: 8px;
      color: white; width: 100%; margin-bottom: 6px;
  }
  [data-testid="stSidebar"] .stButton button:hover { background: #3A7CC4; }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: #F8F9FA; border-radius: 10px;
      padding: 16px; border-left: 4px solid #1E3A5F;
  }

  /* Section headings */
  h2 { color: #1E3A5F; }
  h3 { color: #2E5E9E; }

  /* Tables */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Divider */
  hr { border-top: 2px solid #ECF0F1; }
</style>
""", unsafe_allow_html=True)


# ── Authentication ────────────────────────────────────────────────────────
if not render_login():
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center; font-size:18px;'>🎓 SPPS</h2>"
        "<p style='text-align:center; font-size:11px; opacity:.7;'>"
        "Student Performance<br>Prediction System</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    role     = st.session_state.get("role", "Student")
    username = st.session_state.get("username", "user")
    st.markdown(f"👤 **{username}** `{role}`")
    st.divider()

    # Navigation
    pages = ["🏫 Admin Dashboard", "🎓 Student Analysis"]
    if role == "Student":
        pages = ["🎓 Student Analysis"]

    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.divider()

    # ── Pipeline Controls ────────────────────────────────────────────────
    if role == "Admin":
        st.markdown("**⚙️ Pipeline Controls**")

        if st.button("🔄 Generate Data"):
            with st.spinner("Generating synthetic data…"):
                try:
                    import subprocess, sys
                    result = subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / "generate_data.py")],
                        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                    )
                    if result.returncode == 0:
                        st.success("✅ Data generated!")
                    else:
                        st.error(result.stderr[:300])
                except Exception as e:
                    st.error(str(e))

        if st.button("🧹 Preprocess Data"):
            with st.spinner("Preprocessing…"):
                try:
                    from src.data_preprocessing  import DataPreprocessor
                    from src.feature_engineering import FeatureEngineer
                    dp = DataPreprocessor()
                    df = dp.run()
                    fe = FeatureEngineer()
                    fe.run(df)
                    st.success(f"✅ Preprocessed {len(df)} rows!")
                except Exception as e:
                    st.error(str(e))

        if st.button("🚀 Train Models"):
            with st.spinner("Training models… (may take a minute)"):
                try:
                    from src.train_model import ModelTrainer
                    trainer = ModelTrainer()
                    results = trainer.run()
                    st.success(f"✅ Best model: **{trainer.best_name}**")
                    for m, met in results.items():
                        st.caption(f"{m}: F1={met['f1']:.3f}")
                    st.cache_resource.clear()
                    st.cache_data.clear()
                except Exception as e:
                    st.error(str(e))

        st.divider()

        # ── Batch Prediction from CSV ─────────────────────────────────────
        st.markdown("**📂 Batch Prediction**")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            import pandas as pd
            df_up = pd.read_csv(uploaded)
            st.caption(f"{len(df_up)} rows uploaded.")
            if st.button("⚡ Run Batch Predict"):
                with st.spinner("Predicting…"):
                    try:
                        from src.predict import StudentPredictor
                        predictor = StudentPredictor()
                        df_result = predictor.predict_batch(df_up)
                        st.success("✅ Done!")
                        st.dataframe(df_result[["student_id", "predicted_result",
                                                 "risk_level", "risk_score"]].head(20))
                        csv = df_result.to_csv(index=False).encode()
                        st.download_button("⬇️ Download Results", csv,
                                           "batch_predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(str(e))

        st.divider()

    # ── System Status ─────────────────────────────────────────────────────
    st.markdown("**📋 System Status**")
    model_ok = check_model_files([BEST_MODEL_PATH])
    data_ok  = PROCESSED_DATA_PATH.exists()

    st.markdown(f"{'✅' if model_ok else '❌'} Model trained")
    st.markdown(f"{'✅' if data_ok  else '❌'} Data ready")

    st.divider()
    if st.button("🚪 Logout"):
        logout()


# ── Page Routing ──────────────────────────────────────────────────────────
if "Admin Dashboard" in page:
    render_admin_dashboard()
elif "Student Analysis" in page:
    render_student_dashboard()
