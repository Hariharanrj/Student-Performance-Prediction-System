"""
main.py - Full pipeline runner: generate → preprocess → engineer → train → evaluate.
Run with:  python main.py
"""

import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.utils import setup_logger

logger = setup_logger("Main")


def _banner(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _step(label: str, fn, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    print(f"\n▶  {label}…")
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"   ✅  Done in {elapsed:.1f}s")
        return result
    except Exception as exc:
        logger.exception(f"{label} FAILED")
        print(f"   ❌  {label} failed: {exc}")
        raise


def run_pipeline(
    generate:    bool = True,
    preprocess:  bool = True,
    train:       bool = True,
    evaluate:    bool = True,
    n_students:  int  = 1000,
) -> None:
    """
    Orchestrate the full ML pipeline.

    Args:
        generate:   Whether to (re)generate synthetic data.
        preprocess: Whether to run preprocessing + feature engineering.
        train:      Whether to train / retrain models.
        evaluate:   Whether to run model evaluation after training.
        n_students: Number of synthetic student records to generate.
    """
    _banner("Student Performance Prediction System — Pipeline Start")

    # ── Step 1: Generate Data ─────────────────────────────────────────────
    if generate:
        from generate_data import generate_dataset
        from config.config import RAW_DATA_DIR, SAMPLE_DIR
        from src.utils import save_csv
        import pandas as pd

        def _gen():
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
            df = generate_dataset(n_students)
            save_csv(df, RAW_DATA_DIR / "student_data.csv")
            sample = df.drop(columns=["result"]).sample(min(20, n_students), random_state=42)
            save_csv(sample, SAMPLE_DIR / "sample_students.csv")
            return df

        df_raw = _step(f"Generate {n_students} student records", _gen)
        vc     = df_raw["result"].value_counts()
        print(f"   Pass: {vc.get('Pass',0)}  |  Fail: {vc.get('Fail',0)}")

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    if preprocess:
        from src.data_preprocessing  import DataPreprocessor
        from src.feature_engineering import FeatureEngineer

        dp = DataPreprocessor()
        df_proc = _step("Data Preprocessing", dp.run)

        fe = FeatureEngineer()
        df_eng  = _step("Feature Engineering", fe.run, df_proc)
        print(f"   Final shape: {df_eng.shape}")

    # ── Step 3: Train Models ──────────────────────────────────────────────
    if train:
        from src.train_model import ModelTrainer

        trainer = ModelTrainer()
        results = _step("Model Training", trainer.run)

        print("\n📊  Training Results:")
        for model, m in results.items():
            print(
                f"   {model:25s}  "
                f"Acc={m['accuracy']:.4f}  "
                f"F1={m['f1']:.4f}  "
                f"CV-F1={m['cv_f1_mean']:.4f}"
            )
        print(f"\n🏆  Best Model: {trainer.best_name}")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────
    if evaluate:
        from src.evaluate_model import ModelEvaluator

        ev = ModelEvaluator()

        def _eval():
            ev.load_models()
            ev.prepare_test_data()
            return ev.evaluate_all()

        eval_results = _step("Model Evaluation", _eval)

        print("\n📈  Evaluation Results:")
        for model, m in eval_results.items():
            print(
                f"   {model:25s}  "
                f"Acc={m['accuracy']:.4f}  "
                f"P={m['precision']:.4f}  "
                f"R={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}"
            )

    # ── Done ──────────────────────────────────────────────────────────────
    _banner("Pipeline Complete ✅")
    print("\nNext step → launch the dashboard:")
    print("  streamlit run app/dashboard.py\n")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Student Performance Prediction System — Pipeline Runner"
    )
    parser.add_argument("--no-generate",   action="store_true", help="Skip data generation")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--no-train",      action="store_true", help="Skip model training")
    parser.add_argument("--no-evaluate",   action="store_true", help="Skip model evaluation")
    parser.add_argument("--students",      type=int, default=1000,
                        help="Number of synthetic student records (default: 1000)")
    args = parser.parse_args()

    run_pipeline(
        generate   = not args.no_generate,
        preprocess = not args.no_preprocess,
        train      = not args.no_train,
        evaluate   = not args.no_evaluate,
        n_students = args.students,
    )
