"""
pdf_generator.py - Generates detailed PDF reports for individual students.
Uses ReportLab (pure-Python, no external binaries required).
"""

import sys
import io
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import REPORTS_DIR
from src.utils import setup_logger, get_timestamp

logger = setup_logger("PDFGenerator")

try:
    from reportlab.lib.pagesizes  import letter
    from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units      import inch
    from reportlab.lib.colors     import HexColor, black, white
    from reportlab.platypus       import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums      import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF generation disabled.")


# ── Colour Palette ───────────────────────────────────────────────────────

C_NAVY    = HexColor("#1E3A5F") if REPORTLAB_AVAILABLE else None
C_RED     = HexColor("#E74C3C") if REPORTLAB_AVAILABLE else None
C_ORANGE  = HexColor("#F39C12") if REPORTLAB_AVAILABLE else None
C_GREEN   = HexColor("#27AE60") if REPORTLAB_AVAILABLE else None
C_LIGHT   = HexColor("#ECF0F1") if REPORTLAB_AVAILABLE else None
C_DARK    = HexColor("#2C3E50") if REPORTLAB_AVAILABLE else None


class PDFReportGenerator:
    """
    Generates a styled PDF report for a student.

    Usage:
        gen   = PDFReportGenerator()
        pdf_bytes = gen.generate(student, prediction, recommendations)
        # or save to disk:
        path  = gen.save(student, prediction, recommendations)
    """

    def generate(self,
                 student:         dict,
                 prediction:      dict,
                 recommendations: list[dict]) -> bytes:
        """
        Build and return the PDF as raw bytes (suitable for Streamlit download).

        Args:
            student:         Raw student feature dict.
            prediction:      Prediction result dict.
            recommendations: List of recommendation dicts.

        Returns:
            PDF bytes.
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is not installed. Run: pip install reportlab"
            )

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []
        styles = self._build_styles()

        story += self._header_section(student, prediction, styles)
        story += self._student_details_table(student, styles)
        story += self._prediction_section(prediction, styles)
        story += self._recommendations_section(recommendations, styles)
        story += self._footer_section(styles)

        doc.build(story)
        return buffer.getvalue()

    def save(self,
             student:         dict,
             prediction:      dict,
             recommendations: list[dict]) -> Path:
        """Save PDF to REPORTS_DIR and return the file path."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sid      = student.get("student_id", "UNKNOWN")
        filename = f"report_{sid}_{get_timestamp()}.pdf"
        path     = REPORTS_DIR / filename

        pdf_bytes = self.generate(student, prediction, recommendations)
        path.write_bytes(pdf_bytes)
        logger.info(f"PDF saved → {path}")
        return path

    # ── Private Builders ─────────────────────────────────────────────────

    @staticmethod
    def _build_styles() -> dict:
        base   = getSampleStyleSheet()
        styles = {}

        styles["title"] = ParagraphStyle(
            "Title", parent=base["Title"],
            fontSize=22, fontName="Helvetica-Bold",
            textColor=white, alignment=TA_CENTER, spaceAfter=4,
        )
        styles["subtitle"] = ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=11, fontName="Helvetica",
            textColor=white, alignment=TA_CENTER,
        )
        styles["section"] = ParagraphStyle(
            "Section", parent=base["Heading2"],
            fontSize=13, fontName="Helvetica-Bold",
            textColor=C_NAVY, spaceBefore=14, spaceAfter=6,
        )
        styles["body"] = ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, fontName="Helvetica", spaceAfter=4,
        )
        styles["small"] = ParagraphStyle(
            "Small", parent=base["Normal"],
            fontSize=8, fontName="Helvetica", textColor=HexColor("#888888"),
            alignment=TA_CENTER,
        )
        return styles

    def _header_section(self, student: dict, prediction: dict, styles: dict) -> list:
        risk   = prediction.get("risk_level", "Unknown")
        color  = {"High Risk": C_RED, "Medium Risk": C_ORANGE, "Low Risk": C_GREEN}.get(risk, C_NAVY)

        header_table = Table(
            [[
                Paragraph("🎓 Student Performance Report", styles["title"]),
                Paragraph(
                    f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}<br/>"
                    f"Dept: AI &amp; Data Science",
                    styles["subtitle"],
                ),
            ]],
            colWidths=["60%", "40%"],
        )
        header_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), C_NAVY),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
            ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ]))

        return [header_table, Spacer(1, 12)]

    def _student_details_table(self, student: dict, styles: dict) -> list:
        elements = [Paragraph("📋  Student Information", styles["section"])]

        data = [
            ["Student ID",  student.get("student_id",   "N/A"),
             "Name",        student.get("name",          "N/A")],
            ["Gender",      student.get("gender",        "N/A"),
             "Age",         str(student.get("age",       "N/A"))],
            ["Department",  student.get("department",    "N/A"),
             "—",           ""],
            ["Attendance",  f"{student.get('attendance_percentage', 'N/A')}%",
             "Midterm",     f"{student.get('midterm_marks', 'N/A')}/100"],
            ["Assignment",  f"{student.get('assignment_score', 'N/A')}/100",
             "Quiz",        f"{student.get('quiz_score', 'N/A')}/100"],
            ["Lab Score",   f"{student.get('lab_score', 'N/A')}/100",
             "LMS Activity",f"{student.get('lms_activity', 'N/A')}/100"],
        ]

        tbl = Table(data, colWidths=["22%", "28%", "22%", "28%"])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (0, -1), C_LIGHT),
            ("BACKGROUND",   (2, 0), (2, -1), C_LIGHT),
            ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",     (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("GRID",         (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [white, HexColor("#F9F9F9")]),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        elements += [tbl, Spacer(1, 10)]
        return elements

    def _prediction_section(self, prediction: dict, styles: dict) -> list:
        elements = [Paragraph("🔮  Prediction & Risk Assessment", styles["section"])]

        risk   = prediction.get("risk_level",       "Unknown")
        result = prediction.get("predicted_result", "Unknown")
        score  = prediction.get("risk_score",        0)
        conf   = prediction.get("confidence",         0)

        risk_color = {"High Risk": C_RED, "Medium Risk": C_ORANGE, "Low Risk": C_GREEN}.get(risk, C_NAVY)

        data = [
            ["Metric",          "Value"],
            ["Predicted Result", result],
            ["Risk Level",       risk],
            ["Risk Score",       f"{score:.1f} / 100"],
            ["Model Confidence", f"{conf:.1f}%"],
            ["Pass Probability", f"{prediction.get('pass_probability', 0)*100:.1f}%"],
            ["Fail Probability", f"{prediction.get('fail_probability', 0)*100:.1f}%"],
        ]

        tbl = Table(data, colWidths=["50%", "50%"])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, HexColor("#F9F9F9")]),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        elements += [tbl, Spacer(1, 10)]
        return elements

    def _recommendations_section(self, recs: list[dict], styles: dict) -> list:
        elements = [Paragraph("💡  Personalised Recommendations", styles["section"])]

        priority_color = {
            "CRITICAL": C_RED,
            "HIGH":     C_ORANGE,
            "MEDIUM":   HexColor("#3498DB"),
            "LOW":      C_GREEN,
        }

        for r in recs:
            col = priority_color.get(r.get("priority", "LOW"), C_NAVY)
            elements.append(Paragraph(
                f"<b>{r.get('icon','')} [{r.get('priority','')}] {r.get('category','')}</b> — "
                f"{r.get('message','')}<br/>"
                f"<i>→ Action: {r.get('action','')}</i>",
                styles["body"],
            ))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#EEEEEE")))

        elements.append(Spacer(1, 12))
        return elements

    def _footer_section(self, styles: dict) -> list:
        now = datetime.now().strftime("%d %B %Y")
        return [
            HRFlowable(width="100%", thickness=1, color=C_NAVY),
            Spacer(1, 6),
            Paragraph(
                f"Student Performance Prediction System  |  AI &amp; Data Science Dept  |  {now}<br/>"
                "This report was generated automatically by an ML model and is for academic guidance only.",
                styles["small"],
            ),
        ]


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = PDFReportGenerator()
    demo_student = {
        "student_id": "STU0042", "name": "Priya Kumar",
        "gender": "Female", "age": 20,
        "department": "Artificial Intelligence & Data Science",
        "attendance_percentage": 55.0, "assignment_score": 42.0,
        "quiz_score": 38.0, "lab_score": 48.0,
        "midterm_marks": 35.0, "lms_activity": 25.0,
        "videos_watched": 4,  "discussion_participation": 2,
    }
    demo_prediction = {
        "predicted_result": "Fail", "risk_level": "High Risk",
        "risk_score": 82.5, "confidence": 82.5,
        "pass_probability": 0.175, "fail_probability": 0.825,
    }
    demo_recs = [
        {"icon": "🚨", "priority": "CRITICAL", "category": "Attendance",
         "message": "Attendance critically low at 55%.",
         "action":  "Attend every class immediately."},
        {"icon": "📝", "priority": "HIGH",     "category": "Assignments",
         "message": "Assignment score is 42/100.",
         "action":  "Complete all pending assignments."},
    ]

    path = gen.save(demo_student, demo_prediction, demo_recs)
    print(f"✅  PDF saved → {path}")
