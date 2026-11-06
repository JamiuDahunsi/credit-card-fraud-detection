"""
Exploratory Data Analysis (EDA) module for Credit Card Fraud Detection.

This script inspects the dataset, computes key statistics, analyzes the extreme
class imbalance, and generates publication-quality figures for reports and README.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/creditcard.csv")


def load_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    """Load credit card dataset."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please run the Kaggle download command first."
        )
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded. Shape: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def inspect_basics(df: pd.DataFrame) -> dict:
    """Print and return basic dataset properties."""
    print("\n" + "=" * 60)
    print("1. BASIC DATASET HEALTH CHECK")
    print("=" * 60)
    
    missing_total = df.isnull().sum().sum()
    duplicates_count = df.duplicated().sum()
    
    print(f"Total entries: {len(df):,}")
    print(f"Total features: {df.shape[1]}")
    print(f"Total missing values: {missing_total}")
    print(f"Duplicate rows detected: {duplicates_count:,} ({duplicates_count / len(df):.2%})")
    
    # Class imbalance stats
    class_counts = df["Class"].value_counts()
    n_legit = class_counts.get(0, 0)
    n_fraud = class_counts.get(1, 0)
    fraud_pct = (n_fraud / len(df)) * 100
    
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION (THE IMBALANCE PROBLEM)")
    print("=" * 60)
    print(f"Genuine / Non-Fraud (0): {n_legit:,} ({100 - fraud_pct:.4f}%)")
    print(f"Fraudulent Transactions (1): {n_fraud:,} ({fraud_pct:.4f}%)")
    print(f"Imbalance Ratio: ~1 fraud per {int(round(n_legit / n_fraud)):,} genuine transactions")
    
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_total": missing_total,
        "duplicates": duplicates_count,
        "n_legit": n_legit,
        "n_fraud": n_fraud,
        "fraud_pct": fraud_pct,
    }


def analyze_amount_and_time(df: pd.DataFrame) -> None:
    """Analyze transaction Amount and Time for fraud vs legitimate."""
    print("\n" + "=" * 60)
    print("3. TRANSACTION AMOUNT & TIME METRICS")
    print("=" * 60)
    
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    
    print("\n[Transaction Amount Statistics]")
    stats_df = pd.DataFrame({
        "Genuine ($)": legit["Amount"].describe(),
        "Fraud ($)": fraud["Amount"].describe(),
    })
    print(stats_df.round(2).to_string())
    
    fraud_total_val = fraud["Amount"].sum()
    legit_total_val = legit["Amount"].sum()
    print(f"\nTotal money transacted in dataset: ${df['Amount'].sum():,.2f}")
    print(f"Total fraudulent transaction sum: ${fraud_total_val:,.2f} ({fraud_total_val / df['Amount'].sum():.2%})")
    
    # Time analysis: Time is elapsed seconds from first transaction (over 48 hours)
    duration_hours = df["Time"].max() / 3600
    print(f"\nTotal dataset time span: {duration_hours:.2f} hours (~{duration_hours/24:.1f} days)")


def plot_class_imbalance(df: pd.DataFrame) -> None:
    """Generate and save class imbalance visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df["Class"].value_counts()
    colors = ["#2b5c8f", "#d9534f"]
    labels = ["Genuine (0)", "Fraud (1)"]
    
    # Bar Chart (Log Scale to show minority class clearly)
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_title("Class Frequency (Logarithmic Scale)", weight="bold")
    axes[0].set_ylabel("Count (log scale)")
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    
    # Donut Chart
    axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.2f}%" if pct < 50 else f"{pct:.2f}%",
        explode=(0, 0.2),
        startangle=30,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    axes[1].set_title(f"Class Proportion (Fraud: {counts[1]/len(df):.3%})", weight="bold")
    
    plt.suptitle("Credit Card Fraud Dataset: Extreme Class Imbalance", fontsize=14, weight="bold")
    plt.tight_layout()
    output_path = FIGURES_DIR / "01_class_imbalance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_amount_and_time(df: pd.DataFrame) -> None:
    """Generate Amount and Time comparative distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]
    
    # 1. Amount Histogram (Log Amount)
    sns.histplot(
        np.log1p(legit["Amount"]),
        kde=True,
        color="#2b5c8f",
        label=f"Genuine (n={len(legit):,})",
        ax=axes[0, 0],
        bins=50,
        stat="density",
    )
    sns.histplot(
        np.log1p(fraud["Amount"]),
        kde=True,
        color="#d9534f",
        label=f"Fraud (n={len(fraud):,})",
        ax=axes[0, 0],
        bins=50,
        stat="density",
    )
    axes[0, 0].set_title("Log-Transformed Amount Distribution (log1p($))", weight="bold")
    axes[0, 0].set_xlabel("log(1 + Amount)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    
    # 2. Amount Boxplot (capped for visualization)
    sns.boxplot(
        data=df,
        x="Class",
        y="Amount",
        hue="Class",
        palette=["#2b5c8f", "#d9534f"],
        ax=axes[0, 1],
        showfliers=False,
        legend=False,
    )
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Genuine (0)", "Fraud (1)"])
    axes[0, 1].set_title("Transaction Amount Boxplot (Outliers Omitted for Scale)", weight="bold")
    axes[0, 1].set_ylabel("Transaction Amount ($)")
    
    # 3. Transaction Volume over Time (Hourly)
    df_temp = df.copy()
    df_temp["Hour"] = (df_temp["Time"] / 3600) % 24
    
    sns.kdeplot(
        df_temp[df_temp["Class"] == 0]["Hour"],
        color="#2b5c8f",
        label="Genuine",
        fill=True,
        alpha=0.3,
        ax=axes[1, 0],
    )
    sns.kdeplot(
        df_temp[df_temp["Class"] == 1]["Hour"],
        color="#d9534f",
        label="Fraud",
        fill=True,
        alpha=0.4,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Hourly Distribution of Transactions (24-Hour Cycle)", weight="bold")
    axes[1, 0].set_xlabel("Hour of Day (0 to 24)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlim(0, 24)
    axes[1, 0].legend()
    
    # 4. Scatter: Amount vs Hour
    sample_legit = legit.sample(n=3000, random_state=42)
    sample_df = pd.concat([sample_legit, fraud])
    sample_df["Hour"] = (sample_df["Time"] / 3600) % 24
    
    sns.scatterplot(
        data=sample_df,
        x="Hour",
        y="Amount",
        hue="Class",
        palette=["#2b5c8f", "#d9534f"],
        alpha=0.6,
        s=30,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Amount vs Hour of Day (Fraud vs Sampled Legit)", weight="bold")
    axes[1, 1].set_xlabel("Hour of Day (0 to 24)")
    axes[1, 1].set_ylabel("Amount ($)")
    axes[1, 1].set_ylim(0, 2500)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "02_amount_and_time_analysis.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_correlations(df: pd.DataFrame) -> pd.Series:
    """Identify and plot top correlated features with Class."""
    correlations = df.corr()["Class"].drop("Class").sort_values()
    
    # Top 5 negative and top 5 positive
    top_neg = correlations.head(5)
    top_pos = correlations.tail(5)
    top_features = pd.concat([top_neg, top_pos])
    
    plt.figure(figsize=(10, 6))
    colors = ["#2b5c8f" if val < 0 else "#d9534f" for val in top_features.values]
    bars = plt.barh(top_features.index, top_features.values, color=colors, edgecolor="black")
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Top 5 Negative and Positive Correlated Features with Fraud (Class)", weight="bold")
    plt.xlabel("Pearson Correlation Coefficient")
    
    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.04
        plt.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        
    plt.tight_layout()
    output_path = FIGURES_DIR / "03_feature_correlations.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")
    
    print("\n" + "=" * 60)
    print("4. TOP DISCRIMINATING FEATURES")
    print("=" * 60)
    print("Highest Negative Correlations (High value = Low fraud chance):")
    for feat, corr in top_neg.items():
        print(f"  {feat}: {corr:+.4f}")
    print("\nHighest Positive Correlations (High value = High fraud chance):")
    for feat, corr in top_pos.items():
        print(f"  {feat}: {corr:+.4f}")
        
    return correlations


def plot_top_feature_distributions(df: pd.DataFrame) -> None:
    """Plot boxplots of the top 4 predictive features."""
    top_feats = ["V14", "V17", "V12", "V10"]  # strongest negative correlations
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    palette = {0: "#2b5c8f", 1: "#d9534f"}
    
    for idx, feat in enumerate(top_feats):
        sns.boxplot(
            data=df,
            x="Class",
            y=feat,
            hue="Class",
            palette=palette,
            ax=axes[idx],
            showfliers=False,
            legend=False,
        )
        axes[idx].set_title(f"Distribution of {feat}", weight="bold")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(["Genuine (0)", "Fraud (1)"])
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(f"{feat} Value")
    
    plt.suptitle("Top Predictive PCA Features (Clear Separation Between Fraud & Genuine)", fontsize=14, weight="bold")
    plt.tight_layout()
    output_path = FIGURES_DIR / "04_top_predictive_features_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def run_eda() -> None:
    """Execute complete EDA pipeline."""
    df = load_data()
    inspect_basics(df)
    analyze_amount_and_time(df)
    print("\nGenerating visual figures in reports/figures/...")
    plot_class_imbalance(df)
    plot_amount_and_time(df)
    plot_correlations(df)
    plot_top_feature_distributions(df)
    print("\n[SUCCESS] EDA successfully completed and figures saved!")


if __name__ == "__main__":
    run_eda()
