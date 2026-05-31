# 🎓 Student Performance Prediction System
### Using Machine Learning & Explainable AI

> **B.Tech Final Year Project — Artificial Intelligence & Data Science**  
> An end-to-end ML system that predicts student academic risk, explains model decisions with SHAP, and surfaces actionable interventions through an interactive dashboard.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Quick Start](#-quick-start)
6. [Detailed Setup](#-detailed-setup)
7. [Pipeline Stages](#-pipeline-stages)
8. [Dashboard Guide](#-dashboard-guide)
9. [MySQL Setup](#-mysql-setup)
10. [Deployment](#-deployment)
11. [API Reference](#-api-reference)
12. [Test Suite](#-test-suite)
13. [Demo Credentials](#-demo-credentials)

---

## 🎯 Project Overview

The **Student Performance Prediction System (SPPS)** identifies students at risk of failing their final examinations *before* exams occur — giving faculty time to intervene. It combines three ML classifiers, five engineered features, SHAP-based explainability, a Streamlit dashboard, automated PDF reports, and email alerts.

**Prediction targets:**
| Output | Description |
|--------|-------------|
| Pass / Fail | Binary classification of final exam outcome |
| Risk Score | 0–100 continuous risk index |
| Risk Level | Low / Medium / High categorical label |

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| 🤖 **3 ML Models** | Logistic Regression, Random Forest, XGBoost — auto-selects best by CV F1 |
| 🔍 **Explainable AI** | SHAP waterfall & summary plots per student |
| 📊 **Admin Dashboard** | KPIs, risk distribution, department analytics, confusion matrices, ROC curves |
| 🎓 **Student Dashboard** | Radar chart, gauge, SHAP bar, personalised recommendations |
| 📄 **PDF Reports** | One-click downloadable report per student (ReportLab) |
| 📧 **Email Alerts** | Automatic SMTP alerts for high-risk students |
| 📂 **Batch Prediction** | Upload CSV → predict all → download results |
| 🛢️ **MySQL** | Full schema with views; optional (app works without DB) |
| 🧪 **36 Unit Tests** | Pytest suite covering preprocessing, training, and prediction |

---

## 🛠️ Tech Stack

```
ML:          scikit-learn · XGBoost · SHAP
Frontend:    Streamlit · Plotly · Matplotlib
Database:    MySQL (optional) · mysql-connector-python
PDF:         ReportLab
Language:    Python 3.10+
```

---

## 📁 Project Structure

```
Student-Performance-Prediction-System/
├── app/
│   ├── dashboard.py          ← Streamlit entry point (run this)
│   ├── admin_dashboard.py    ← Admin overview & model metrics
│   ├── student_dashboard.py  ← Per-student prediction & SHAP
│   └── login.py              ← Role-based auth
│
├── config/
│   ├── config.py             ← All paths, thresholds, DB config
│   └── settings.json         ← JSON settings (logging, UI, etc.)
│
├── data/
│   ├── raw/                  ← Raw CSV (generated or uploaded)
│   ├── processed/            ← Preprocessed + feature-engineered data
│   └── sample/               ← 20-row sample for batch-predict demo
│
├── database/
│   ├── schema.sql            ← MySQL DDL (tables + views)
│   ├── insert_data.sql       ← Sample INSERT statements
│   └── db_connection.py      ← DatabaseManager class
│
├── models/
│   ├── trained/              ← Saved model .pkl files
│   └── scaler/               ← StandardScaler .pkl
│
├── reports/generated_reports/ ← PDF output directory
│
├── src/
│   ├── data_preprocessing.py  ← DataPreprocessor class
│   ├── feature_engineering.py ← FeatureEngineer class (5 features)
│   ├── train_model.py         ← ModelTrainer class
│   ├── evaluate_model.py      ← ModelEvaluator class
│   ├── predict.py             ← StudentPredictor (single & batch)
│   ├── recommendation_engine.py ← Rule-based recommendations
│   ├── explainability.py      ← SHAP ExplainabilityEngine
│   ├── email_alert.py         ← EmailAlertSystem (SMTP)
│   ├── pdf_generator.py       ← PDFReportGenerator (ReportLab)
│   └── utils.py               ← Logging, I/O helpers
│
├── tests/
│   ├── test_preprocessing.py  ← 12 unit tests
│   ├── test_training.py       ← 13 unit tests
│   └── test_prediction.py     ← 11 unit tests
│
├── generate_data.py           ← Synthetic dataset generator
├── main.py                    ← Full pipeline CLI
├── run_project.py             ← One-click setup + launch
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/Student-Performance-Prediction-System.git
cd Student-Performance-Prediction-System

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline + launch dashboard
python run_project.py
```

The dashboard opens at **http://localhost:8501**

---

## 🔧 Detailed Setup

### Step-by-step (manual)

```bash
# Generate 1000 synthetic student records
python generate_data.py

# Preprocess + feature engineering
python -c "
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
df = DataPreprocessor().run()
FeatureEngineer().run(df)
"

# Train all 3 models (saves best automatically)
python src/train_model.py

# Evaluate models
python src/evaluate_model.py

# Launch dashboard
streamlit run app/dashboard.py
```

### Environment Variables (optional)

Create a `.env` file for email and database:

```dotenv
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=student_performance_db

# Email alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=admin@college.edu
```

---

## 🔄 Pipeline Stages

```
generate_data.py
    ↓  1000 synthetic students (correlated features)
DataPreprocessor.run()
    ↓  Label encode, impute NaN, cap outliers, encode Pass/Fail → 0/1
FeatureEngineer.run()
    ↓  +5 engineered features (attendance_risk, engagement_score, …)
ModelTrainer.run()
    ↓  Train LR + RF + XGBoost, cross-validate, save best model
ModelEvaluator.evaluate_all()
    ↓  Accuracy / Precision / Recall / F1 / AUC / Confusion Matrix
StudentPredictor.predict_single() / predict_batch()
    ↓  Risk score + risk level + probabilities
RecommendationEngine.generate()
    ↓  Priority-ranked, rule-based action items
ExplainabilityEngine.waterfall_plot()
    ↓  SHAP feature contributions per student
PDFReportGenerator.generate()
    ↓  Downloadable PDF report
```

---

## 🖥️ Dashboard Guide

### Admin Dashboard
| Section | Description |
|---------|-------------|
| KPI Cards | Total students · High-risk count · Avg attendance · Pass rate |
| Attendance Distribution | Histogram with 75% threshold line |
| Risk Distribution | Pie chart (High / Medium / Low) |
| Performance Scatter | Attendance vs Midterm, coloured by risk |
| Dept. Performance | Bar chart of predicted pass rate per department |
| Model Metrics | Side-by-side table + bar chart + ROC curves |
| Confusion Matrix | Selectable per model |
| High-Risk Table | Filtered, sorted student list |

### Student Dashboard
| Section | Description |
|---------|-------------|
| Input Method | Manual form OR select from dataset |
| Result Banner | Pass/Fail · Risk Level · Confidence |
| Risk Gauge | 0–100 speedometer chart |
| Radar Chart | 7-axis academic profile vs 75% target |
| Recommendations | Expandable cards sorted by priority |
| SHAP Bar Chart | Feature contributions (green = Pass, red = Fail) |
| PDF Download | One-click ReportLab-generated PDF |

### Sidebar Controls (Admin only)
- 🔄 **Generate Data** — re-generate synthetic dataset
- 🧹 **Preprocess Data** — run full preprocessing pipeline  
- 🚀 **Train Models** — retrain all 3 models
- 📂 **Batch Prediction** — upload CSV, download predictions

---

## 🛢️ MySQL Setup

```bash
# Create schema
mysql -u root -p < database/schema.sql

# (Optional) Insert sample data
mysql -u root -p student_performance_db < database/insert_data.sql
```

Update `config/config.py` with your credentials or set environment variables.

**Tables created:**
- `students` — raw student records
- `predictions` — model prediction history
- `model_metrics` — training run metrics
- `alert_logs` — email alert history

**Views created:**
- `v_student_predictions` — latest prediction per student
- `v_risk_summary` — risk level counts and averages

---

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub (ensure models are committed or regenerated on deploy)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file: `app/dashboard.py`
4. Add secrets in the Streamlit Cloud dashboard (DB, email)

### Render

```yaml
# render.yaml
services:
  - type: web
    name: spps
    env: python
    buildCommand: "pip install -r requirements.txt && python main.py --no-generate"
    startCommand: "streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0"
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python main.py
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address", "0.0.0.0"]
```

---

## 📐 API Reference

### `StudentPredictor`
```python
from src.predict import StudentPredictor
p = StudentPredictor()

# Single prediction
result = p.predict_single(record_dict)
# → {'predicted_result', 'risk_level', 'risk_score', 'confidence', ...}

# Batch prediction
df_out = p.predict_batch(df)
# → original df + prediction columns appended
```

### `RecommendationEngine`
```python
from src.recommendation_engine import RecommendationEngine
recs = RecommendationEngine().generate(record, "High Risk")
# → [{'category', 'priority', 'icon', 'message', 'action'}, ...]
```

### `ExplainabilityEngine`
```python
from src.explainability import ExplainabilityEngine
eng = ExplainabilityEngine()
eng.load_model()
fig  = eng.waterfall_plot(X_instance, feature_names)
fig2 = eng.summary_plot(X_background)
df   = eng.feature_importance_df(X_background)
```

### `PDFReportGenerator`
```python
from src.pdf_generator import PDFReportGenerator
gen = PDFReportGenerator()
pdf_bytes = gen.generate(student, prediction, recommendations)
path      = gen.save(student, prediction, recommendations)
```

---

## 🧪 Test Suite

```bash
# Run all 36 tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_training.py -v
python -m pytest tests/test_prediction.py -v
```

**Test coverage:**
| Module | Tests |
|--------|-------|
| `DataPreprocessor` | 6 tests — NaN imputation, outlier capping, encoding, dedup |
| `FeatureEngineer` | 6 tests — all engineered features, bounds, NaN-free |
| `ModelTrainer` | 6 tests — model count, metric ranges, best model selection |
| `ModelEvaluator` | 5 tests — metrics, CM figure, ROC figure, DataFrame |
| `StudentPredictor` | 8 tests — keys, result values, probabilities, risk |
| `RecommendationEngine` | 5 tests — structure, priority, formatting |

---

## 🔑 Demo Credentials

| Role | Username | Password | Access |
|------|----------|----------|--------|
| Admin | `admin` | `admin123` | Full access + pipeline controls |
| Faculty | `faculty` | `faculty123` | Dashboards only |
| Student | `student` | `student123` | Student analysis only |

Click **"Demo (Admin)"** on the login page for instant access.

---

## 📊 Dataset Features

| Category | Features |
|----------|---------|
| **Student Info** | student_id, name, gender, age, department |
| **Academic** | attendance_percentage, assignment_score, quiz_score, lab_score, midterm_marks |
| **Engagement** | lms_activity, videos_watched, discussion_participation |
| **Engineered** | attendance_risk, assignment_completion_ratio, engagement_score, performance_trend, overall_academic_score |
| **Target** | result (Pass=1 / Fail=0) |

---

## 👥 Authors

**AI & Data Science Department — B.Tech Final Year Project**

---

## 📄 License

This project is developed for academic purposes as a B.Tech Final Year Project in Artificial Intelligence & Data Science.
