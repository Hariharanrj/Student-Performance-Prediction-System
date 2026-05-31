"""
email_alert.py - Sends automated email alerts for high-risk students.
"""

import sys
import smtplib
import logging
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import EMAIL_CONFIG
from src.utils import setup_logger

logger = setup_logger("EmailAlert")


class EmailAlertSystem:
    """
    Sends HTML email alerts for high-risk students via SMTP.

    Usage:
        alert = EmailAlertSystem()
        alert.send_high_risk_alert(student_record, prediction_result)
        alert.send_batch_alerts(high_risk_df)
    """

    def __init__(self):
        self.cfg = EMAIL_CONFIG

    # ── Public API ──────────────────────────────────────────────────────

    def send_high_risk_alert(self, student: dict, prediction: dict) -> bool:
        """
        Send a single high-risk alert email.

        Args:
            student:    Raw student record dict.
            prediction: Prediction result dict from StudentPredictor.

        Returns:
            True if sent successfully, False otherwise.
        """
        subject = (
            f"🚨 HIGH RISK ALERT — {student.get('name', 'Student')} "
            f"({student.get('student_id', 'N/A')})"
        )
        html    = self._build_alert_html(student, prediction)
        return self._send(subject, html)

    def send_batch_alerts(self, high_risk_records: list[dict]) -> dict:
        """
        Send alerts for all high-risk students in a list.

        Returns:
            dict with 'sent' and 'failed' counts.
        """
        results = {"sent": 0, "failed": 0}
        for record in high_risk_records:
            ok = self.send_high_risk_alert(record, record)
            if ok:
                results["sent"]   += 1
            else:
                results["failed"] += 1

        logger.info(
            f"Batch alerts: {results['sent']} sent, {results['failed']} failed."
        )
        return results

    def test_connection(self) -> bool:
        """Verify SMTP credentials without sending a message."""
        try:
            with smtplib.SMTP(self.cfg["smtp_server"], self.cfg["smtp_port"], timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self.cfg["sender"], self.cfg["password"])
            logger.info("SMTP connection test passed.")
            return True
        except Exception as exc:
            logger.error(f"SMTP connection test failed: {exc}")
            return False

    # ── Private Helpers ──────────────────────────────────────────────────

    def _send(self, subject: str, html_body: str) -> bool:
        """Low-level send via SMTP with TLS."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self.cfg["sender"]
        msg["To"]      = self.cfg["receiver"]
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(self.cfg["smtp_server"], self.cfg["smtp_port"], timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.cfg["sender"], self.cfg["password"])
                server.sendmail(self.cfg["sender"], self.cfg["receiver"], msg.as_string())
            logger.info(f"Email sent: {subject[:60]}")
            return True
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed. Check EMAIL_SENDER / EMAIL_PASSWORD.")
            return False
        except smtplib.SMTPException as exc:
            logger.error(f"SMTP error: {exc}")
            return False
        except Exception as exc:
            logger.error(f"Unexpected email error: {exc}")
            return False

    @staticmethod
    def _build_alert_html(student: dict, prediction: dict) -> str:
        """Build a styled HTML email body."""
        now   = datetime.now().strftime("%d %b %Y, %H:%M")
        risk  = prediction.get("risk_level",       "High Risk")
        score = prediction.get("risk_score",        "N/A")
        result= prediction.get("predicted_result", "N/A")
        conf  = prediction.get("confidence",        "N/A")

        return f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body      {{ font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px; }}
    .container{{ max-width: 600px; margin: auto; background: #fff;
                 border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.15); }}
    .header   {{ background: #C0392B; color: #fff; padding: 20px; text-align: center; }}
    .body     {{ padding: 25px; }}
    .badge    {{ display: inline-block; background: #E74C3C; color: #fff;
                 padding: 4px 12px; border-radius: 20px; font-weight: bold; }}
    table     {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
    th,td     {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th        {{ background: #F5F5F5; font-weight: bold; }}
    .footer   {{ background: #2C3E50; color: #aaa; padding: 15px; text-align: center;
                 font-size: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>🚨 Student Performance Alert</h2>
      <p>Automated notification — {now}</p>
    </div>
    <div class="body">
      <p>Dear Faculty / Advisor,</p>
      <p>The Student Performance Prediction System has identified the following student
         as <span class="badge">HIGH RISK</span> of failing their final examination.</p>

      <table>
        <tr><th>Student ID</th><td>{student.get('student_id','N/A')}</td></tr>
        <tr><th>Name</th>      <td>{student.get('name','N/A')}</td></tr>
        <tr><th>Department</th><td>{student.get('department','N/A')}</td></tr>
        <tr><th>Attendance</th><td>{student.get('attendance_percentage','N/A')}%</td></tr>
        <tr><th>Midterm Marks</th><td>{student.get('midterm_marks','N/A')}/100</td></tr>
        <tr><th>Risk Level</th><td><strong style="color:#C0392B;">{risk}</strong></td></tr>
        <tr><th>Risk Score</th><td>{score}/100</td></tr>
        <tr><th>Predicted Result</th><td>{result}</td></tr>
        <tr><th>Model Confidence</th><td>{conf}%</td></tr>
      </table>

      <p style="margin-top:20px;">
        <strong>Recommended Actions:</strong>
      </p>
      <ul>
        <li>Schedule a one-on-one counselling session with the student.</li>
        <li>Enrol the student in remedial / catch-up classes.</li>
        <li>Monitor attendance closely over the next two weeks.</li>
        <li>Inform parents / guardians if necessary.</li>
      </ul>

      <p style="color:#888; font-size:12px;">
        This alert was generated automatically by the SPPS ML System.<br>
        Please do not reply to this email.
      </p>
    </div>
    <div class="footer">
      Student Performance Prediction System &copy; {datetime.now().year} — AI &amp; Data Science Dept.
    </div>
  </div>
</body>
</html>
"""


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    alert = EmailAlertSystem()
    ok = alert.test_connection()
    print("SMTP connection:", "✅ OK" if ok else "❌ Failed")
