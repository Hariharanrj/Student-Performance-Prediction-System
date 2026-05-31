-- ============================================================
-- schema.sql  –  Student Performance Prediction System DB
-- Run once:  mysql -u root -p < database/schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS student_performance_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE student_performance_db;

-- ── students ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS students (
    id                       INT            AUTO_INCREMENT PRIMARY KEY,
    student_id               VARCHAR(20)    NOT NULL UNIQUE,
    name                     VARCHAR(100)   NOT NULL,
    gender                   VARCHAR(10)    NOT NULL,
    age                      TINYINT        NOT NULL,
    department               VARCHAR(100)   NOT NULL,

    -- Academic features
    attendance_percentage    DECIMAL(5,2)   NOT NULL DEFAULT 0.00,
    assignment_score         DECIMAL(5,2)   NOT NULL DEFAULT 0.00,
    quiz_score               DECIMAL(5,2)   NOT NULL DEFAULT 0.00,
    lab_score                DECIMAL(5,2)   NOT NULL DEFAULT 0.00,
    midterm_marks            DECIMAL(5,2)   NOT NULL DEFAULT 0.00,

    -- Engagement features
    lms_activity             DECIMAL(5,2)   NOT NULL DEFAULT 0.00,
    videos_watched           SMALLINT       NOT NULL DEFAULT 0,
    discussion_participation SMALLINT       NOT NULL DEFAULT 0,

    -- Ground-truth label
    result                   ENUM('Pass','Fail') DEFAULT NULL,

    -- Timestamps
    created_at               TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    updated_at               TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_department (department),
    INDEX idx_result     (result)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ── predictions ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id                INT            AUTO_INCREMENT PRIMARY KEY,
    student_id        VARCHAR(20)    NOT NULL,
    predicted_result  ENUM('Pass','Fail') NOT NULL,
    risk_level        ENUM('High Risk','Medium Risk','Low Risk') NOT NULL,
    risk_score        DECIMAL(5,2)   NOT NULL,
    pass_probability  DECIMAL(6,4)   NOT NULL,
    fail_probability  DECIMAL(6,4)   NOT NULL,
    confidence        DECIMAL(5,2)   NOT NULL,
    predicted_at      TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_student_id  (student_id),
    INDEX idx_risk_level  (risk_level),
    INDEX idx_predicted_at(predicted_at),

    CONSTRAINT fk_pred_student
        FOREIGN KEY (student_id)
        REFERENCES  students(student_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ── model_metrics ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_metrics (
    id          INT           AUTO_INCREMENT PRIMARY KEY,
    model_name  VARCHAR(50)   NOT NULL,
    accuracy    DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall      DECIMAL(6,4),
    f1_score    DECIMAL(6,4),
    auc         DECIMAL(6,4),
    is_best     TINYINT(1)    DEFAULT 0,
    trained_at  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_model_name (model_name),
    INDEX idx_trained_at (trained_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ── alert_logs ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS alert_logs (
    id          INT           AUTO_INCREMENT PRIMARY KEY,
    student_id  VARCHAR(20)   NOT NULL,
    alert_type  VARCHAR(50)   DEFAULT 'HIGH_RISK_EMAIL',
    sent_to     VARCHAR(200),
    status      ENUM('sent','failed','pending') DEFAULT 'pending',
    sent_at     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_student_id (student_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ── Useful views ─────────────────────────────────────────────

CREATE OR REPLACE VIEW v_student_predictions AS
SELECT
    s.student_id,
    s.name,
    s.department,
    s.attendance_percentage,
    s.midterm_marks,
    s.result                    AS actual_result,
    p.predicted_result,
    p.risk_level,
    p.risk_score,
    p.confidence,
    p.predicted_at
FROM students s
LEFT JOIN (
    SELECT * FROM predictions p1
    WHERE predicted_at = (
        SELECT MAX(p2.predicted_at)
        FROM predictions p2
        WHERE p2.student_id = p1.student_id
    )
) p ON s.student_id = p.student_id;


CREATE OR REPLACE VIEW v_risk_summary AS
SELECT
    risk_level,
    COUNT(*) AS student_count,
    AVG(risk_score) AS avg_risk_score
FROM (
    SELECT student_id, risk_level, risk_score,
           ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY predicted_at DESC) AS rn
    FROM predictions
) latest
WHERE rn = 1
GROUP BY risk_level;


SELECT 'Schema created successfully.' AS status;
