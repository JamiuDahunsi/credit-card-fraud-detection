"""
Credit Card Fraud Detection - Exploratory Data Analysis (EDA)

This script loads the creditcard.csv dataset, checks basic stats,
and plots the class distribution, transaction amounts, and time patterns.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

# 1. Load dataset
data_path = Path("data/creditcard.csv")
if not data_path.exists():
    raise FileNotFoundError(
        "creditcard.csv not found in data/ folder. "
        "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )

print("Loading data...")
df = pd.read_csv(data_path)
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 2. Check missing values & class counts
print(f"Missing values: {df.isnull().sum().sum()}")
class_counts = df["Class"].value_counts()
n_legit = class_counts[0]
n_fraud = class_counts[1]
fraud_pct = (n_fraud / len(df)) * 100

print(f"Legitimate transactions (0): {n_legit:,} ({100 - fraud_pct:.3f}%)")
print(f"Fraudulent transactions (1): {n_fraud:,} ({fraud_pct:.3f}%)")
print(f"Imbalance ratio: 1 fraud per ~{int(round(n_legit / n_fraud))} legitimate transactions\n")

# 3. Transaction Amount stats
print("--- Amount Statistics ($) ---")
print("Legitimate:")
print(df[df["Class"] == 0]["Amount"].describe().round(2))
print("\nFraudulent:")
print(df[df["Class"] == 1]["Amount"].describe().round(2))

# 4. Plot Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].bar(["Genuine (0)", "Fraud (1)"], [n_legit, n_fraud], color=["#2b5c8f", "#d9534f"])
axes[0].set_yscale("log")
axes[0].set_title("Transaction Count (Log Scale)", fontweight="bold")
axes[0].set_ylabel("Count")

axes[1].pie(
    [n_legit, n_fraud],
    labels=["Genuine (99.83%)", "Fraud (0.17%)"],
    colors=["#2b5c8f", "#d9534f"],
    explode=(0, 0.2),
    startangle=30,
)
axes[1].set_title("Class Imbalance Ratio", fontweight="bold")

plt.tight_layout()
plt.savefig(output_dir / "class_imbalance.png", dpi=300)
plt.close()
print(f"Saved: {output_dir / 'class_imbalance.png'}")

# 5. Plot Amount & Time distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log-amount distribution
legit_amt = df[df["Class"] == 0]["Amount"]
fraud_amt = df[df["Class"] == 1]["Amount"]

sns.histplot(np.log1p(legit_amt), bins=40, color="#2b5c8f", label="Genuine", stat="density", ax=axes[0])
sns.histplot(np.log1p(fraud_amt), bins=40, color="#d9534f", label="Fraud", stat="density", ax=axes[0])
axes[0].set_title("Transaction Amount Distribution (log1p scale)", fontweight="bold")
axes[0].set_xlabel("log(1 + Amount)")
axes[0].legend()

# Hourly pattern (seconds -> hour of day)
df["Hour"] = (df["Time"] / 3600) % 24
sns.kdeplot(df[df["Class"] == 0]["Hour"], color="#2b5c8f", label="Genuine", fill=True, alpha=0.3, ax=axes[1])
sns.kdeplot(df[df["Class"] == 1]["Hour"], color="#d9534f", label="Fraud", fill=True, alpha=0.4, ax=axes[1])
axes[1].set_title("Transaction Volume by Hour of Day", fontweight="bold")
axes[1].set_xlabel("Hour (0 - 24)")
axes[1].set_xlim(0, 24)
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / "amount_and_time.png", dpi=300)
plt.close()
print(f"Saved: {output_dir / 'amount_and_time.png'}")

print("\nEDA completed successfully!")
