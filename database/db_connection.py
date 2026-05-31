"""
db_connection.py - MySQL database connection and CRUD operations.
"""

import sys
import logging
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DB_CONFIG
from src.utils import setup_logger

logger = setup_logger("Database")

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("mysql-connector-python not installed. DB features disabled.")


class DatabaseManager:
    """
    Wraps MySQL connection with CRUD helpers.

    Usage:
        db = DatabaseManager()
        db.connect()
        students = db.fetch_all_students()
        db.insert_prediction(student_id, prediction_dict)
        db.close()
    """

    def __init__(self, config: dict | None = None):
        self.config = config or DB_CONFIG
        self._conn  = None

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open a connection. Returns True on success."""
        if not MYSQL_AVAILABLE:
            logger.error("mysql-connector-python is not installed.")
            return False
        try:
            self._conn = mysql.connector.connect(**self.config)
            if self._conn.is_connected():
                logger.info(f"Connected to MySQL: {self.config['database']} @ {self.config['host']}")
                return True
        except MySQLError as e:
            logger.error(f"MySQL connection error: {e}")
        return False

    def close(self) -> None:
        if self._conn and self._conn.is_connected():
            self._conn.close()
            logger.info("MySQL connection closed.")

    def is_connected(self) -> bool:
        return bool(self._conn and self._conn.is_connected())

    @contextmanager
    def cursor(self, dictionary: bool = True):
        """Context manager that yields a cursor and auto-commits."""
        if not self.is_connected():
            self.connect()
        cur = self._conn.cursor(dictionary=dictionary)
        try:
            yield cur
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            logger.error(f"DB error — rolled back: {exc}")
            raise
        finally:
            cur.close()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def execute(self, query: str, params: tuple = ()) -> bool:
        """Run a DML statement (INSERT / UPDATE / DELETE)."""
        try:
            with self.cursor() as cur:
                cur.execute(query, params)
            return True
        except Exception as exc:
            logger.error(f"execute() error: {exc}")
            return False

    def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Run a SELECT and return all rows as list-of-dicts."""
        try:
            with self.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall() or []
        except Exception as exc:
            logger.error(f"fetch_all() error: {exc}")
            return []

    def fetch_dataframe(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Run a SELECT and return a DataFrame."""
        rows = self.fetch_all(query, params)
        return pd.DataFrame(rows)

    # ── Domain-specific helpers ──────────────────────────────────────────

    def fetch_all_students(self) -> pd.DataFrame:
        return self.fetch_dataframe("SELECT * FROM students ORDER BY student_id")

    def fetch_student_by_id(self, student_id: str) -> dict | None:
        rows = self.fetch_all(
            "SELECT * FROM students WHERE student_id = %s LIMIT 1",
            (student_id,),
        )
        return rows[0] if rows else None

    def insert_student(self, record: dict) -> bool:
        sql = """
        INSERT INTO students
            (student_id, name, gender, age, department,
             attendance_percentage, assignment_score, quiz_score,
             lab_score, midterm_marks, lms_activity,
             videos_watched, discussion_participation, result)
        VALUES
            (%(student_id)s, %(name)s, %(gender)s, %(age)s, %(department)s,
             %(attendance_percentage)s, %(assignment_score)s, %(quiz_score)s,
             %(lab_score)s, %(midterm_marks)s, %(lms_activity)s,
             %(videos_watched)s, %(discussion_participation)s, %(result)s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name), attendance_percentage=VALUES(attendance_percentage)
        """
        try:
            with self.cursor() as cur:
                cur.execute(sql, record)
            logger.info(f"Inserted/updated student {record.get('student_id')}")
            return True
        except Exception as exc:
            logger.error(f"insert_student error: {exc}")
            return False

    def insert_prediction(self, student_id: str, prediction: dict) -> bool:
        sql = """
        INSERT INTO predictions
            (student_id, predicted_result, risk_level, risk_score,
             pass_probability, fail_probability, confidence, predicted_at)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        params = (
            student_id,
            prediction.get("predicted_result"),
            prediction.get("risk_level"),
            prediction.get("risk_score"),
            prediction.get("pass_probability"),
            prediction.get("fail_probability"),
            prediction.get("confidence"),
        )
        return self.execute(sql, params)

    def fetch_predictions(self, student_id: str | None = None) -> pd.DataFrame:
        if student_id:
            return self.fetch_dataframe(
                "SELECT * FROM predictions WHERE student_id = %s ORDER BY predicted_at DESC",
                (student_id,),
            )
        return self.fetch_dataframe(
            "SELECT * FROM predictions ORDER BY predicted_at DESC LIMIT 500"
        )

    def get_dashboard_stats(self) -> dict:
        """Return aggregate stats for the admin dashboard."""
        stats = {}
        try:
            rows = self.fetch_all("""
                SELECT
                  COUNT(*)                                       AS total_students,
                  AVG(attendance_percentage)                     AS avg_attendance,
                  SUM(result = 'Pass')                           AS pass_count,
                  SUM(result = 'Fail')                           AS fail_count
                FROM students
            """)
            if rows:
                row = rows[0]
                stats["total_students"]  = row.get("total_students", 0)
                stats["avg_attendance"]  = round(float(row.get("avg_attendance") or 0), 2)
                stats["pass_count"]      = row.get("pass_count", 0)
                stats["fail_count"]      = row.get("fail_count", 0)

            high_risk = self.fetch_all("""
                SELECT COUNT(DISTINCT student_id) AS cnt
                FROM predictions
                WHERE risk_level = 'High Risk'
            """)
            stats["high_risk_count"] = high_risk[0].get("cnt", 0) if high_risk else 0

        except Exception as exc:
            logger.error(f"get_dashboard_stats error: {exc}")
        return stats

    def bulk_insert_students(self, df: pd.DataFrame) -> int:
        """Insert multiple student rows; return number successfully inserted."""
        inserted = 0
        for _, row in df.iterrows():
            record = row.to_dict()
            if self.insert_student(record):
                inserted += 1
        logger.info(f"Bulk insert: {inserted}/{len(df)} rows.")
        return inserted


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    db = DatabaseManager()
    ok = db.connect()
    if ok:
        stats = db.get_dashboard_stats()
        print("Dashboard stats:", stats)
        db.close()
    else:
        print("❌  Could not connect to MySQL. Check config/config.py settings.")
