"""
Credit Card Fraud Detection - Model Training & Evaluation

This script trains baseline and ensemble models, compares their ability to
catch fraud (Recall) vs. avoid false alarms (Precision), and saves evaluation charts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
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
import xgboost as xgb

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Train / Test Split (Stratified to keep the 0.17% fraud ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train):,} ({y_train.sum()} frauds)")
print(f"Testing samples:  {len(X_test):,} ({y_test.sum()} frauds)")

# 3. Scale 'Amount' and 'Time' using RobustScaler (resilient to large outliers)
# Fit on train only to prevent data leakage
scaler_amount = RobustScaler()
scaler_time = RobustScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train["scaled_amount"] = scaler_amount.fit_transform(X_train[["Amount"]])
X_train["scaled_time"] = scaler_time.fit_transform(X_train[["Time"]])
X_train.drop(["Time", "Amount"], axis=1, inplace=True)

X_test["scaled_amount"] = scaler_amount.transform(X_test[["Amount"]])
X_test["scaled_time"] = scaler_time.transform(X_test[["Time"]])
X_test.drop(["Time", "Amount"], axis=1, inplace=True)

# 4. Define Models to Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest (Balanced)": RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    ),
}

# 5. Train & Evaluate
results = {}
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
cm_plots = {}

print("\n" + "=" * 70)
print(f"{'Model':<26} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<6}")
print("=" * 70)

for name, clf in models.items():
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    cm_plots[name] = cm
    results[name] = {"PR-AUC": pr_auc, "ROC-AUC": roc_auc, "Recall": rec, "Precision": prec, "F1": f1}
    
    print(f"{name:<26} | {pr_auc:<8.4f} | {roc_auc:<8.4f} | {rec:<8.2%} | {prec:<9.2%} | {f1:<6.4f}")
    
    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ax_pr.plot(recall_vals, precision_vals, label=f"{name} (PR-AUC = {pr_auc:.3f})", linewidth=2)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.3f})", linewidth=2)

print("=" * 70)

# Save PR & ROC Curves side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for name, clf in models.items():
    y_prob = clf.predict_proba(X_test)[:, 1]
    p, r, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    axes[0].plot(r, p, label=f"{name} (PR-AUC: {average_precision_score(y_test, y_prob):.3f})", linewidth=2)
    axes[1].plot(fpr, tpr, label=f"{name} (ROC-AUC: {roc_auc_score(y_test, y_prob):.3f})", linewidth=2)

axes[0].set_title("Precision-Recall Curve (Primary Benchmark)", fontweight="bold")
axes[0].set_xlabel("Recall (Fraud Catch Rate)")
axes[0].set_ylabel("Precision (Accuracy of Flags)")
axes[0].legend(loc="lower left")

axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_title("ROC Curve", fontweight="bold")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(images_dir / "pr_and_roc_curves.png", dpi=300)
plt.close()
print(f"Saved: {images_dir / 'pr_and_roc_curves.png'}")

# Save Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for idx, (name, cm) in enumerate(cm_plots.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx], annot_kws={"size": 12, "weight": "bold"})
    axes[idx].set_title(name, fontweight="bold")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
    axes[idx].set_xticklabels(["Legit", "Fraud"])
    axes[idx].set_yticklabels(["Legit", "Fraud"])

plt.suptitle("Confusion Matrices (Test Set: 56,962 Transactions, 98 Frauds)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(images_dir / "confusion_matrices.png", dpi=300)
plt.close()
print(f"Saved: {images_dir / 'confusion_matrices.png'}")

print("\nModel evaluation completed!")
