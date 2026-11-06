"""
Main execution pipeline for Credit Card Fraud Detection.

Runs end-to-end EDA, model training, benchmarking, and evaluation visualization.
Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.eda import run_eda
from src.model import train_and_evaluate_all
from src.evaluate import run_evaluation


def main():
    print("=" * 80)
    print("   CREDIT CARD FRAUD DETECTION END-TO-END PIPELINE")
    print("=" * 80)
    
    print("\n>>> STAGE 1: Exploratory Data Analysis & Feature Insights...")
    run_eda()
    
    print("\n>>> STAGE 2: Model Training & Imbalance Strategy Benchmark...")
    train_and_evaluate_all()
    
    print("\n>>> STAGE 3: Metric Evaluation, Curves & Financial Cost Simulation...")
    run_evaluation()
    
    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully!")
    print("All figures saved to 'reports/figures/'.")
    print("Best model weights saved to 'models/best_fraud_model.joblib'.")
    print("=" * 80)


if __name__ == "__main__":
    main()
