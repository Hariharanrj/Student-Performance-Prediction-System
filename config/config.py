"""
config.py - Central configuration for Student Performance Prediction System
Author: AI & Data Science Final Year Project
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR        = BASE_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
SAMPLE_DIR      = DATA_DIR / "sample"

MODELS_DIR      = BASE_DIR / "models"
TRAINED_DIR     = MODELS_DIR / "trained"
SCALER_DIR      = MODELS_DIR / "scaler"

REPORTS_DIR     = BASE_DIR / "reports" / "generated_reports"
LOGS_DIR        = BASE_DIR / "logs"
ASSETS_DIR      = BASE_DIR / "assets"

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────
RAW_DATA_PATH       = RAW_DATA_DIR / "student_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_student_data.csv"
SAMPLE_DATA_PATH    = SAMPLE_DIR / "sample_students.csv"

BEST_MODEL_PATH     = TRAINED_DIR / "best_model.pkl"
RF_MODEL_PATH       = TRAINED_DIR / "random_forest.pkl"
LR_MODEL_PATH       = TRAINED_DIR / "logistic_regression.pkl"
XGB_MODEL_PATH      = TRAINED_DIR / "xgboost.pkl"
SCALER_PATH         = SCALER_DIR  / "scaler.pkl"

LOG_FILE_PATH       = LOGS_DIR / "app.log"

# ─────────────────────────────────────────────
# Database Configuration
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME",     "student_performance_db"),
    "charset":  "utf8mb4",
    "autocommit": True,
}

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────
MODEL_CONFIG = {
    "test_size":        0.2,
    "random_state":     42,
    "cv_folds":         5,
    "scoring_metric":   "f1",

    "random_forest": {
        "n_estimators":   200,
        "max_depth":      10,
        "min_samples_split": 5,
        "random_state":   42,
        "n_jobs":         -1,
    },

    "logistic_regression": {
        "max_iter":       1000,
        "random_state":   42,
        "C":              1.0,
    },

    "xgboost": {
        "n_estimators":   200,
        "max_depth":      6,
        "learning_rate":  0.1,
        "random_state":   42,
        "eval_metric":    "logloss",
        "use_label_encoder": False,
    },
}

# ─────────────────────────────────────────────
# Risk Classification Thresholds
# ─────────────────────────────────────────────
RISK_THRESHOLDS = {
    "high":   0.70,   # probability of failing >= 0.70  → High Risk
    "medium": 0.40,   # probability of failing >= 0.40  → Medium Risk
                      # below 0.40                       → Low Risk
}

RISK_LABELS = {
    "high":   "High Risk",
    "medium": "Medium Risk",
    "low":    "Low Risk",
}

RISK_COLORS = {
    "High Risk":   "#FF4B4B",
    "Medium Risk": "#FFA500",
    "Low Risk":    "#00CC44",
}

# ─────────────────────────────────────────────
# Feature Lists
# ─────────────────────────────────────────────
CATEGORICAL_FEATURES = ["gender", "department"]

NUMERIC_FEATURES = [
    "age", "attendance_percentage",
    "assignment_score", "quiz_score", "lab_score",
    "midterm_marks", "lms_activity", "videos_watched",
    "discussion_participation",
]

ENGINEERED_FEATURES = [
    "attendance_risk", "assignment_completion_ratio",
    "engagement_score", "performance_trend", "overall_academic_score",
]

TARGET_COLUMN   = "result"
ID_COLUMN       = "student_id"
NAME_COLUMN     = "name"

# All features used for model training (after encoding)
ALL_MODEL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES

# ─────────────────────────────────────────────
# Email Configuration
# ─────────────────────────────────────────────
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port":   int(os.getenv("SMTP_PORT", "587")),
    "sender":      os.getenv("EMAIL_SENDER", "your_email@gmail.com"),
    "password":    os.getenv("EMAIL_PASSWORD", "your_app_password"),
    "receiver":    os.getenv("EMAIL_RECEIVER", "admin@college.edu"),
}

# ─────────────────────────────────────────────
# UI Configuration
# ─────────────────────────────────────────────
UI_CONFIG = {
    "page_title":  "Student Performance Prediction System",
    "page_icon":   "🎓",
    "layout":      "wide",
    "theme_color": "#1E3A5F",
}

# ─────────────────────────────────────────────
# Department List
# ─────────────────────────────────────────────
DEPARTMENTS = [
    "Artificial Intelligence & Data Science",
    "Computer Science Engineering",
    "Electronics & Communication",
    "Mechanical Engineering",
    "Civil Engineering",
    "Information Technology",
]

GENDERS = ["Male", "Female", "Other"]
