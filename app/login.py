"""
login.py - Simple role-based login for the SPPS Streamlit app.
"""

import hashlib
import streamlit as st

# ── Credential Store (hash passwords in production) ───────────────────────

USERS = {
    "admin":   {"password_hash": hashlib.sha256(b"admin123").hexdigest(),  "role": "Admin"},
    "faculty": {"password_hash": hashlib.sha256(b"faculty123").hexdigest(),"role": "Faculty"},
    "student": {"password_hash": hashlib.sha256(b"student123").hexdigest(),"role": "Student"},
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def render_login() -> bool:
    """
    Display the login UI.
    Returns True when the user is authenticated (stored in st.session_state).
    """
    # ── Already logged in ─────────────────────────────────────────────────
    if st.session_state.get("authenticated"):
        return True

    # ── Login card ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding:30px 0 10px'>
          <h1 style='color:#1E3A5F;'>🎓 Student Performance Prediction System</h1>
          <p style='color:#666; font-size:16px;'>
            AI-powered academic risk assessment & early intervention platform
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        with st.container(border=True):
            st.subheader("🔐 Login")
            username = st.text_input("Username", placeholder="admin / faculty / student")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            remember = st.checkbox("Remember me")

            col1, col2 = st.columns(2)
            with col1:
                login_btn = st.button("Login", type="primary", use_container_width=True)
            with col2:
                demo_btn  = st.button("Demo (Admin)", use_container_width=True)

        if login_btn:
            user = USERS.get(username.lower())
            if user and user["password_hash"] == _hash(password):
                _set_session(username, user["role"])
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

        if demo_btn:
            _set_session("admin", "Admin")
            st.rerun()

        st.caption(
            "Demo credentials: admin / admin123  |  faculty / faculty123  |  student / student123"
        )

    # ── Feature highlights ────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.info("🤖 ML Predictions\nLogistic Regression, Random Forest, XGBoost")
    c2.info("🔍 Explainable AI\nSHAP waterfall & feature importance")
    c3.info("📊 Analytics\nInteractive dashboards & charts")
    c4.info("📄 Reports\nAuto-generated downloadable PDF")

    return False


def _set_session(username: str, role: str) -> None:
    st.session_state["authenticated"] = True
    st.session_state["username"]      = username
    st.session_state["role"]          = role


def logout() -> None:
    for key in ["authenticated", "username", "role"]:
        st.session_state.pop(key, None)
    st.rerun()
