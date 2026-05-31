"""
generate_data.py - Generates a realistic synthetic student dataset for the SPPS project.
Run once:  python generate_data.py
"""

import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import RAW_DATA_DIR, SAMPLE_DIR, DEPARTMENTS

SEED        = 42
N_STUDENTS  = 1000
N_SAMPLE    = 20

random.seed(SEED)
np.random.seed(SEED)

FIRST_NAMES = [
    "Aarav", "Aisha", "Arjun", "Priya", "Ravi", "Sneha", "Vikram", "Ananya",
    "Karthik", "Divya", "Rohit", "Meera", "Suresh", "Lakshmi", "Deepak",
    "Kavya", "Mohan", "Pooja", "Sanjay", "Nisha", "Rahul", "Shreya",
    "Amit", "Swati", "Vijay", "Rani", "Ramesh", "Geeta", "Arun", "Sunita",
    "Naveen", "Rekha", "Ajay", "Usha", "Manoj", "Saranya", "Dinesh", "Geetha",
]

LAST_NAMES = [
    "Kumar", "Sharma", "Patel", "Reddy", "Nair", "Menon", "Iyer", "Pillai",
    "Gupta", "Singh", "Rao", "Mehta", "Bose", "Das", "Joshi", "Chandra",
    "Verma", "Mishra", "Yadav", "Pandey", "Tiwari", "Dubey", "Shah", "Modi",
]


def _generate_one(student_id: int) -> dict:
    """Generate one student record with correlated realistic values."""
    gender = random.choice(["Male", "Female", "Other"])
    department = random.choice(DEPARTMENTS)
    age = random.randint(18, 24)

    # Core academic attributes — slightly correlated
    base_ability = np.random.beta(2, 2)          # 0-1, centred around 0.5

    attendance   = np.clip(base_ability * 80 + np.random.normal(10, 10), 30, 100)
    assignment   = np.clip(base_ability * 70 + np.random.normal(15, 12), 10, 100)
    quiz         = np.clip(base_ability * 70 + np.random.normal(10, 12), 5,  100)
    lab          = np.clip(base_ability * 75 + np.random.normal(12, 10), 10, 100)
    midterm      = np.clip(base_ability * 60 + np.random.normal(10, 10), 5,  100)

    # Engagement features
    lms_activity = np.clip(base_ability * 80 + np.random.normal(10, 15), 0, 100)
    videos       = np.clip(base_ability * 50 + np.random.normal(5,  10), 0, 60)
    discussion   = np.clip(base_ability * 40 + np.random.normal(5,  8),  0, 50)

    # Determine result using a weighted rule
    score = (
        attendance   * 0.25 +
        assignment   * 0.20 +
        midterm      * 0.25 +
        quiz         * 0.15 +
        lab          * 0.15
    )
    fail_prob = 1 / (1 + np.exp(0.1 * (score - 55)))   # logistic
    result    = "Fail" if random.random() < fail_prob else "Pass"

    return {
        "student_id":               f"STU{student_id:04d}",
        "name":                     f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "gender":                   gender,
        "age":                      age,
        "department":               department,
        "attendance_percentage":    round(attendance,   2),
        "assignment_score":         round(assignment,   2),
        "quiz_score":               round(quiz,         2),
        "lab_score":                round(lab,          2),
        "midterm_marks":            round(midterm,      2),
        "lms_activity":             round(lms_activity, 2),
        "videos_watched":           int(videos),
        "discussion_participation": int(discussion),
        "result":                   result,
    }


def generate_dataset(n: int = N_STUDENTS) -> pd.DataFrame:
    records = [_generate_one(i + 1) for i in range(n)]
    return pd.DataFrame(records)


if __name__ == "__main__":
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(N_STUDENTS)
    out_raw = RAW_DATA_DIR / "student_data.csv"
    df.to_csv(out_raw, index=False)
    print(f"✅  Generated {len(df)} student records → {out_raw}")

    # Class distribution
    vc = df["result"].value_counts()
    print(f"   Pass: {vc.get('Pass', 0)}  |  Fail: {vc.get('Fail', 0)}")

    # Sample CSV (no result column — for batch-prediction demo)
    sample = df.drop(columns=["result"]).sample(N_SAMPLE, random_state=SEED)
    out_sample = SAMPLE_DIR / "sample_students.csv"
    sample.to_csv(out_sample, index=False)
    print(f"✅  Generated {N_SAMPLE}-row sample CSV → {out_sample}")
