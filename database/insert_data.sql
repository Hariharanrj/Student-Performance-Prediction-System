-- insert_data.sql — Sample student records for testing
USE student_performance_db;

INSERT INTO students (student_id, name, gender, age, department,
  attendance_percentage, assignment_score, quiz_score, lab_score,
  midterm_marks, lms_activity, videos_watched, discussion_participation, result)
VALUES
('STU0001','Aarav Kumar','Male',20,'Artificial Intelligence & Data Science',
  92.5,88.0,84.0,90.0,79.0,87.0,42,28,'Pass'),
('STU0002','Priya Sharma','Female',19,'Computer Science Engineering',
  45.0,35.0,30.0,40.0,28.0,22.0,5,2,'Fail'),
('STU0003','Ravi Reddy','Male',21,'Electronics & Communication',
  78.0,65.0,60.0,70.0,55.0,58.0,20,10,'Pass'),
('STU0004','Ananya Nair','Female',20,'Artificial Intelligence & Data Science',
  55.0,42.0,38.0,48.0,35.0,30.0,8,3,'Fail'),
('STU0005','Karthik Iyer','Male',22,'Mechanical Engineering',
  85.0,75.0,70.0,80.0,68.0,72.0,30,15,'Pass')
ON DUPLICATE KEY UPDATE name=VALUES(name);

INSERT INTO predictions (student_id, predicted_result, risk_level, risk_score,
  pass_probability, fail_probability, confidence)
VALUES
('STU0001','Pass','Low Risk',5.20,0.9480,0.0520,94.80),
('STU0002','Fail','High Risk',87.40,0.1260,0.8740,87.40),
('STU0003','Pass','Low Risk',22.10,0.7790,0.2210,77.90),
('STU0004','Fail','High Risk',78.60,0.2140,0.7860,78.60),
('STU0005','Pass','Medium Risk',41.50,0.5850,0.4150,58.50);

INSERT INTO model_metrics (model_name, accuracy, precision_score, recall, f1_score, auc, is_best)
VALUES
('Logistic Regression', 0.7700, 0.7012, 0.6211, 0.6588, 0.8241, 1),
('Random Forest',       0.7650, 0.6944, 0.6122, 0.6508, 0.8197, 0),
('XGBoost',             0.7500, 0.6788, 0.5900, 0.6314, 0.8050, 0);

SELECT 'Sample data inserted successfully.' AS status;
