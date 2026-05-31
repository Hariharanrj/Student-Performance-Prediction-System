"""
run_project.py - One-click setup: pipeline → launch Streamlit dashboard.
Run with:  python run_project.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("  🎓 Student Performance Prediction System")
    print("  Full Setup & Launch")
    print("=" * 60)

    # ── Run the ML pipeline ───────────────────────────────────────────────
    print("\n[1/2]  Running ML pipeline…")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py")],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("❌  Pipeline failed. Check logs/app.log for details.")
        sys.exit(result.returncode)

    # ── Launch Streamlit ──────────────────────────────────────────────────
    print("\n[2/2]  Launching Streamlit dashboard…")
    print("       URL: http://localhost:8501\n")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "app" / "dashboard.py"),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
        ],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
