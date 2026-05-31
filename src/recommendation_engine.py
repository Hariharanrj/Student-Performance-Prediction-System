"""
recommendation_engine.py - Generates personalised, rule-based academic recommendations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger

logger = setup_logger("RecommendationEngine")


class RecommendationEngine:
    """
    Generates a list of prioritised, personalised recommendations for a student
    based on their feature values and predicted risk level.

    Usage:
        engine = RecommendationEngine()
        recs   = engine.generate(student_record, risk_level)
    """

    # ── Thresholds ───────────────────────────────────────────────────────
    THRESHOLDS = {
        "attendance_low":       75,
        "attendance_critical":  60,
        "assignment_low":       50,
        "quiz_low":             45,
        "lab_low":              50,
        "midterm_low":          40,
        "lms_low":              30,
        "videos_low":           10,
        "discussion_low":        5,
    }

    # ── Priority Levels ──────────────────────────────────────────────────
    PRIORITY = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}

    def generate(self, record: dict, risk_level: str = "Low Risk") -> list[dict]:
        """
        Generate recommendations for a student.

        Args:
            record:     dict of raw feature values.
            risk_level: "High Risk" / "Medium Risk" / "Low Risk".

        Returns:
            Sorted list of recommendation dicts with keys:
              category, message, priority, icon, action.
        """
        recs = []
        T    = self.THRESHOLDS
        att  = float(record.get("attendance_percentage", 100))
        asn  = float(record.get("assignment_score",      100))
        quiz = float(record.get("quiz_score",            100))
        lab  = float(record.get("lab_score",             100))
        mid  = float(record.get("midterm_marks",         100))
        lms  = float(record.get("lms_activity",          100))
        vid  = float(record.get("videos_watched",         20))
        disc = float(record.get("discussion_participation", 10))

        # ── Attendance ───────────────────────────────────────────────────
        if att < T["attendance_critical"]:
            recs.append(self._rec(
                "Attendance", "CRITICAL", "🚨",
                f"Your attendance is critically low ({att:.1f}%). "
                "You are at serious risk of being barred from exams. "
                "Attend every class without exception.",
                "Attend all remaining classes immediately.",
            ))
        elif att < T["attendance_low"]:
            recs.append(self._rec(
                "Attendance", "HIGH", "⚠️",
                f"Attendance is below the required 75% threshold ({att:.1f}%). "
                "Missing further classes may lead to grade penalties.",
                "Aim for 100% attendance in remaining weeks.",
            ))

        # ── Assignments ──────────────────────────────────────────────────
        if asn < T["assignment_low"]:
            recs.append(self._rec(
                "Assignments", "HIGH", "📝",
                f"Assignment score ({asn:.1f}/100) is low. "
                "Incomplete assignments significantly lower your overall grade.",
                "Complete all pending assignments and seek faculty feedback.",
            ))

        # ── Quizzes ──────────────────────────────────────────────────────
        if quiz < T["quiz_low"]:
            recs.append(self._rec(
                "Quiz Performance", "MEDIUM", "🧩",
                f"Quiz score ({quiz:.1f}/100) indicates gaps in understanding. "
                "Regular self-testing can improve retention.",
                "Practice past quizzes and solve textbook exercises daily.",
            ))

        # ── Lab / Practicals ─────────────────────────────────────────────
        if lab < T["lab_low"]:
            recs.append(self._rec(
                "Lab / Practicals", "MEDIUM", "🔬",
                f"Lab score ({lab:.1f}/100) is weak. "
                "Practical skills are weighted heavily in AI/DS programs.",
                "Spend extra hours in the lab and complete all practicals.",
            ))

        # ── Midterm ──────────────────────────────────────────────────────
        if mid < T["midterm_low"]:
            recs.append(self._rec(
                "Midterm Exam", "HIGH", "📊",
                f"Midterm marks ({mid:.1f}/100) are below the pass threshold. "
                "This strongly predicts difficulty in finals.",
                "Attend remedial classes and form study groups for revision.",
            ))

        # ── LMS Activity ─────────────────────────────────────────────────
        if lms < T["lms_low"]:
            recs.append(self._rec(
                "LMS Engagement", "MEDIUM", "💻",
                f"LMS activity score ({lms:.1f}/100) is very low. "
                "Online resources and assignments may have been missed.",
                "Log in to the LMS daily and complete all digital tasks.",
            ))

        # ── Video Learning ───────────────────────────────────────────────
        if vid < T["videos_low"]:
            recs.append(self._rec(
                "Video Learning", "LOW", "🎥",
                f"Only {int(vid)} lecture videos watched. "
                "Video content reinforces classroom learning.",
                "Watch at least 2 lecture videos per day to catch up.",
            ))

        # ── Discussion ───────────────────────────────────────────────────
        if disc < T["discussion_low"]:
            recs.append(self._rec(
                "Peer Discussion", "LOW", "💬",
                f"Discussion participation is very low ({int(disc)} interactions). "
                "Collaborative learning improves problem-solving skills.",
                "Participate in at least one forum thread per week.",
            ))

        # ── Risk-level overarching recommendations ───────────────────────
        if risk_level == "High Risk":
            recs.append(self._rec(
                "Remedial Support", "CRITICAL", "🏫",
                "You are classified as HIGH RISK. Immediate intervention is recommended.",
                "Meet with your faculty advisor and register for remedial classes this week.",
            ))
        elif risk_level == "Medium Risk":
            recs.append(self._rec(
                "Academic Support", "HIGH", "📚",
                "You are at MEDIUM RISK. Consistent effort now can prevent failure.",
                "Create a structured study plan and track weekly progress.",
            ))
        else:
            recs.append(self._rec(
                "Maintain Performance", "LOW", "⭐",
                "Great work — you are LOW RISK! Keep up the momentum.",
                "Continue current habits and help peers who are struggling.",
            ))

        # ── Sort by priority ─────────────────────────────────────────────
        recs.sort(key=lambda r: self.PRIORITY.get(r["priority"], 99))
        logger.info(f"Generated {len(recs)} recommendations (risk={risk_level}).")
        return recs

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _rec(category: str, priority: str, icon: str,
             message: str, action: str) -> dict:
        return {
            "category": category,
            "priority": priority,
            "icon":     icon,
            "message":  message,
            "action":   action,
        }

    def format_for_display(self, recs: list[dict]) -> list[str]:
        """Return a list of formatted strings suitable for Streamlit display."""
        lines = []
        for r in recs:
            lines.append(
                f"{r['icon']} **[{r['priority']}] {r['category']}** — "
                f"{r['message']} *→ {r['action']}*"
            )
        return lines


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = RecommendationEngine()
    demo   = {
        "attendance_percentage":    55.0,
        "assignment_score":         42.0,
        "quiz_score":               38.0,
        "lab_score":                48.0,
        "midterm_marks":            35.0,
        "lms_activity":             25.0,
        "videos_watched":           4,
        "discussion_participation": 2,
    }
    recs = engine.generate(demo, "High Risk")
    print(f"\n📋  {len(recs)} Recommendations:")
    for r in recs:
        print(f"  [{r['priority']:8s}] {r['icon']} {r['category']}: {r['action']}")
