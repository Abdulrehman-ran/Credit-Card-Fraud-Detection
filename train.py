"""
╔══════════════════════════════════════════════════════════════╗
║         CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE     ║
║                                                              ║
║  This script orchestrates the complete ML pipeline:          ║
║    1. Data Loading & Inspection                              ║
║    2. Data Cleaning & Preprocessing                          ║
║    3. Exploratory Data Analysis (EDA)                        ║
║    4. Feature Scaling & SMOTE Balancing                      ║
║    5. Model Training (LR, DT, RF, XGBoost)                  ║
║    6. Evaluation & Comparison                                ║
║    7. Best Model Selection & Saving                          ║
║                                                              ║
║  Usage:                                                      ║
║    python train.py                                           ║
║                                                              ║
║  Prerequisites:                                              ║
║    - Place creditcard.csv in the dataset/ folder             ║
║    - Install requirements: pip install -r requirements.txt   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    setup_logging,
    create_directories,
    format_metrics,
    save_object,
    save_results,
)
from src.data_preprocessing import preprocess_pipeline
from src.eda import run_full_eda
from src.model_training import train_and_evaluate_all


def main():
    """
    Main function — runs the complete fraud detection pipeline.
    """
    # ──────────────────────────────────────────
    #  Setup
    # ──────────────────────────────────────────
    logger = setup_logging()
    start_time = time.time()
    
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE        ║")
    logger.info("╚" + "═" * 58 + "╝")
    
    # Create directories
    dirs = create_directories(PROJECT_ROOT)
    
    # Dataset path
    dataset_path = os.path.join(dirs["dataset"], "creditcard.csv")
    
    # ──────────────────────────────────────────
    #  Step 1: Preprocessing Pipeline
    # ──────────────────────────────────────────
    logger.info("\n📦 STEP 1: Data Preprocessing")
    logger.info("─" * 40)
    
    data = preprocess_pipeline(dataset_path)
    
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    scaler = data["scaler"]
    original_df = data["original_df"]
    
    # ──────────────────────────────────────────
    #  Step 2: Exploratory Data Analysis
    # ──────────────────────────────────────────
    logger.info("\n📊 STEP 2: Exploratory Data Analysis")
    logger.info("─" * 40)
    
    insights = run_full_eda(original_df)
    
    # ──────────────────────────────────────────
    #  Step 3: Model Training & Evaluation
    # ──────────────────────────────────────────
    logger.info("\n🤖 STEP 3: Model Training & Evaluation")
    logger.info("─" * 40)
    
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    # Print comparison table
    logger.info("\n📋 MODEL COMPARISON TABLE:")
    print(format_metrics(results["all_results"]))
    
    # ──────────────────────────────────────────
    #  Step 4: Save Best Model & Artifacts
    # ──────────────────────────────────────────
    logger.info("\n💾 STEP 4: Saving Best Model & Artifacts")
    logger.info("─" * 40)
    
    best_name = results["best_model_name"]
    best_model = results["best_model"]
    
    # Save the best model
    model_path = os.path.join(dirs["models"], "best_model.pkl")
    save_object(best_model, model_path)
    logger.info(f"  Best model ({best_name}) saved to: {model_path}")
    
    # Save the scaler (needed for prediction in the app)
    scaler_path = os.path.join(dirs["models"], "scaler.pkl")
    save_object(scaler, scaler_path)
    logger.info(f"  Scaler saved to: {scaler_path}")
    
    # Save all results as JSON (for the Streamlit dashboard)
    results_data = {
        "best_model_name": best_name,
        "best_metrics": results["best_metrics"],
        "all_results": results["all_results"],
        "feature_names": data["feature_names"],
        "dataset_summary": data["summary"],
        "insights": insights,
    }
    results_path = os.path.join(dirs["models"], "training_results.json")
    save_results(results_data, results_path)
    logger.info(f"  Training results saved to: {results_path}")
    
    # ──────────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────────
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  🏆 Best Model:   {best_name}")
    logger.info(f"  📈 F1 Score:     {results['best_metrics']['F1 Score']:.4f}")
    logger.info(f"  📈 ROC-AUC:      {results['best_metrics']['ROC-AUC']:.4f}")
    logger.info(f"  📈 Accuracy:     {results['best_metrics']['Accuracy']:.4f}")
    logger.info(f"  ⏱  Time Elapsed: {elapsed:.1f} seconds")
    logger.info(f"  📁 Model saved:  {model_path}")
    logger.info(f"  📁 Scaler saved: {scaler_path}")
    logger.info("=" * 60)
    logger.info("\n🚀 Next step: Run the Streamlit dashboard:")
    logger.info("   streamlit run app.py")
    

if __name__ == "__main__":
    main()
