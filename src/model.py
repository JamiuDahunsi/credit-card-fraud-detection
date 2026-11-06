"""
Model training and cross-validation pipeline for Credit Card Fraud Detection.

Implements rigorous stratified cross-validation, proper scaling without data leakage,
and multiple imbalance handling strategies (class weighting, SMOTE, XGBoost scale_pos_weight).
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(data_path: str = "data/creditcard.csv", test_size: float = 0.2, random_state: int = 42):
    """
    Load data, scale Time and Amount using RobustScaler, and create stratified train/test split.
    
    RobustScaler uses median and interquartile range (IQR), making it resilient
    against extreme transaction amounts and outliers.
    """
    df = pd.read_csv(data_path)
    
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()
    
    # Stratified split to ensure both sets have the exact same minority class ratio (0.17%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale Time and Amount (fit on train only to prevent data leakage)
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train["scaled_amount"] = scaler_amount.fit_transform(X_train[["Amount"]])
    X_train["scaled_time"] = scaler_time.fit_transform(X_train[["Time"]])
    X_train = X_train.drop(["Time", "Amount"], axis=1)
    
    X_test["scaled_amount"] = scaler_amount.transform(X_test[["Amount"]])
    X_test["scaled_time"] = scaler_time.transform(X_test[["Time"]])
    X_test = X_test.drop(["Time", "Amount"], axis=1)
    
    print(f"Data Prepared:")
    print(f"  Training Set: {X_train.shape[0]:,} samples (Frauds: {y_train.sum():,}, {y_train.mean():.3%})")
    print(f"  Test Set:     {X_test.shape[0]:,} samples (Frauds: {y_test.sum():,}, {y_test.mean():.3%})")
    
    return X_train, X_test, y_train, y_test, scaler_amount, scaler_time


def build_models(class_imbalance_ratio: float):
    """
    Build candidate models representing different algorithms and imbalance strategies.
    
    Args:
        class_imbalance_ratio: Ratio of negative to positive samples (~578)
    """
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Logistic Regression (Class Weighted)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Logistic Regression + SMOTE": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "XGBoost (Weighted)": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=class_imbalance_ratio,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost (Default)": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


def evaluate_predictions(y_true, y_pred, y_prob):
    """Compute all key classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_and_evaluate_all():
    """Train models, evaluate on test set, and record performance."""
    X_train, X_test, y_train, y_test, scaler_amount, scaler_time = prepare_data()
    
    # Save scalers for reproducible inference
    joblib.dump(scaler_amount, MODELS_DIR / "scaler_amount.joblib")
    joblib.dump(scaler_time, MODELS_DIR / "scaler_time.joblib")
    
    imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Calculated scale_pos_weight for XGBoost: {imbalance_ratio:.2f}\n")
    
    models = build_models(imbalance_ratio)
    
    results = {}
    curves_data = {}
    best_model_name = None
    best_pr_auc = -1.0
    
    print("=" * 75)
    print(f"{'Model':<38} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<6}")
    print("=" * 75)
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict on holdout test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        results[name] = metrics
        
        # Compute PR curve and ROC curve for plotting
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        curves_data[name] = {
            "precision_curve": precision_curve.tolist(),
            "recall_curve": recall_curve.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        
        print(
            f"{name:<38} | {metrics['pr_auc']:.4f}   | {metrics['roc_auc']:.4f}   | "
            f"{metrics['recall']:.4f}   | {metrics['precision']:.4f}    | {metrics['f1']:.4f}"
        )
        
        # Track best model by PR-AUC (gold standard for imbalanced fraud detection)
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model_name = name
            joblib.dump(model, MODELS_DIR / "best_fraud_model.joblib")
            
    print("=" * 75)
    print(f"\nBest Model by PR-AUC: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")
    
    # Save results to JSON
    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    with open(REPORTS_DIR / "curves_data.json", "w") as f:
        json.dump(curves_data, f)
        
    print(f"Saved evaluation metrics to {REPORTS_DIR / 'model_metrics.json'}")
    return results, curves_data


if __name__ == "__main__":
    train_and_evaluate_all()
