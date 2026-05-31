"""
utils.py - Shared utility functions for Student Performance Prediction System
"""

import os
import sys
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import LOG_FILE_PATH, LOGS_DIR


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

def setup_logger(name: str = "SPPS", level: str = "INFO") -> logging.Logger:
    """
    Configure and return a rotating-file + console logger.

    Args:
        name:  Logger name shown in log entries.
        level: Logging level string (DEBUG / INFO / WARNING / ERROR).

    Returns:
        Configured Logger instance.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler (10 MB, 5 backups)
    fh = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ─────────────────────────────────────────────
# File I/O Helpers
# ─────────────────────────────────────────────

def save_pickle(obj, path: Path) -> None:
    """Serialize an object to a pickle file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle → {path}")


def load_pickle(path: Path):
    """Load and return a pickled object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle ← {path}")
    return obj


def load_json(path: Path) -> dict:
    """Load and return a JSON file as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with logging."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"Loaded CSV ← {path}  shape={df.shape}")
    return df


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to CSV with logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Saved CSV → {path}  shape={df.shape}")


# ─────────────────────────────────────────────
# Data Validation
# ─────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, required_cols: list) -> tuple[bool, list]:
    """
    Check that all required columns exist in the DataFrame.

    Returns:
        (True, []) if valid; (False, [missing cols]) otherwise.
    """
    missing = [c for c in required_cols if c not in df.columns]
    return (len(missing) == 0), missing


def validate_numeric_range(value: float, col_name: str,
                            min_val: float = 0, max_val: float = 100) -> float:
    """Clamp a numeric value to [min_val, max_val] and warn if out of range."""
    if value < min_val or value > max_val:
        logger.warning(f"{col_name}={value} is outside [{min_val}, {max_val}]; clamped.")
        return float(np.clip(value, min_val, max_val))
    return float(value)


# ─────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────

def format_percentage(value: float, decimals: int = 1) -> str:
    """Return value formatted as a percentage string, e.g. '87.5%'."""
    return f"{value:.{decimals}f}%"


def format_score(value: float, decimals: int = 2) -> str:
    """Return value formatted as a fixed-decimal string."""
    return f"{value:.{decimals}f}"


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return the current timestamp as a formatted string."""
    return datetime.now().strftime(fmt)


def get_report_filename(student_id: str) -> str:
    """Generate a unique PDF report filename for a student."""
    ts = get_timestamp()
    return f"report_{student_id}_{ts}.pdf"


# ─────────────────────────────────────────────
# Risk / Result Helpers
# ─────────────────────────────────────────────

def risk_color(risk_level: str) -> str:
    """Return a hex color for a risk label."""
    mapping = {
        "High Risk":   "#FF4B4B",
        "Medium Risk": "#FFA500",
        "Low Risk":    "#00CC44",
    }
    return mapping.get(risk_level, "#AAAAAA")


def risk_emoji(risk_level: str) -> str:
    """Return an emoji for a risk label."""
    mapping = {
        "High Risk":   "🔴",
        "Medium Risk": "🟡",
        "Low Risk":    "🟢",
    }
    return mapping.get(risk_level, "⚪")


def result_emoji(result: str) -> str:
    """Return an emoji for a Pass/Fail result."""
    return "✅" if str(result).lower() == "pass" else "❌"


# ─────────────────────────────────────────────
# DataFrame Display Helpers
# ─────────────────────────────────────────────

def style_risk_column(val: str) -> str:
    """Pandas Styler function — color-code the Risk Level column."""
    colors = {
        "High Risk":   "background-color:#FF4B4B; color:white; font-weight:bold",
        "Medium Risk": "background-color:#FFA500; color:white; font-weight:bold",
        "Low Risk":    "background-color:#00CC44; color:white; font-weight:bold",
    }
    return colors.get(val, "")


def display_metric_card(label: str, value, delta=None) -> dict:
    """Return a dict suitable for st.metric()."""
    return {"label": label, "value": value, "delta": delta}


# ─────────────────────────────────────────────
# Environment / Dependency Checks
# ─────────────────────────────────────────────

def check_model_files(paths: list) -> bool:
    """Return True only if all model files exist."""
    for p in paths:
        if not Path(p).exists():
            logger.warning(f"Model file missing: {p}")
            return False
    return True


def ensure_dirs(*dirs) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
