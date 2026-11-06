"""
Model evaluation, comparison visualization, and cost-benefit analysis module.

Generates high-resolution Precision-Recall curves, ROC curves, confusion matrices,
and financial fraud loss comparisons.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def load_results():
    """Load cached metrics and curve coordinates."""
    with open(REPORTS_DIR / "model_metrics.json", "r") as f:
        metrics = json.load(f)
    with open(REPORTS_DIR / "curves_data.json", "r") as f:
        curves = json.load(f)
    return metrics, curves


def plot_pr_and_roc_curves(metrics, curves):
    """Plot Precision-Recall and ROC curves side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
    ]
    
    # 1. Precision-Recall Curve (Crucial for Imbalanced Data)
    for idx, (name, data) in enumerate(curves.items()):
        pr_auc = metrics[name]["pr_auc"]
        axes[0].plot(
            data["recall_curve"],
            data["precision_curve"],
            label=f"{name} (PR-AUC = {pr_auc:.3f})",
            linewidth=2,
            color=palette[idx % len(palette)],
        )
        
    axes[0].set_title("Precision-Recall (PR) Curves\n(Primary Benchmark for Fraud Imbalance)", weight="bold", fontsize=13)
    axes[0].set_xlabel("Recall (Fraud Catch Rate)", fontsize=11)
    axes[0].set_ylabel("Precision (True Fraud / Flagged)", fontsize=11)
    axes[0].set_xlim([0.0, 1.02])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].legend(loc="lower left", fontsize=9, frameon=True)
    
    # 2. ROC Curve
    for idx, (name, data) in enumerate(curves.items()):
        roc_auc = metrics[name]["roc_auc"]
        axes[1].plot(
            data["fpr"],
            data["tpr"],
            label=f"{name} (ROC-AUC = {roc_auc:.3f})",
            linewidth=2,
            color=palette[idx % len(palette)],
        )
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Guess (0.50)")
    axes[1].set_title("Receiver Operating Characteristic (ROC) Curves", weight="bold", fontsize=13)
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[1].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[1].set_xlim([-0.02, 1.0])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "05_pr_and_roc_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_metrics_comparison_bar(metrics):
    """Bar chart comparing Precision, Recall, F1, PR-AUC, and ROC-AUC."""
    df_metrics = pd.DataFrame(metrics).T
    plot_df = df_metrics[["pr_auc", "roc_auc", "f1", "recall", "precision"]]
    plot_df.columns = ["PR-AUC", "ROC-AUC", "F1-Score", "Recall", "Precision"]
    
    ax = plot_df.plot(
        kind="bar",
        figsize=(15, 6),
        colormap="viridis",
        edgecolor="black",
        width=0.8,
    )
    plt.title("Model Performance Metrics Comparison Across Imbalance Strategies", weight="bold", fontsize=14)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Model Configuration", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", frameon=True)
    
    # Annotate bar values
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
                
    plt.tight_layout()
    output_path = FIGURES_DIR / "06_metrics_comparison_bar.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_confusion_matrices(metrics):
    """Plot confusion matrices for all evaluated models in a grid."""
    models = list(metrics.keys())
    n_models = len(models)
    
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten()
    
    for idx, name in enumerate(models):
        m = metrics[name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{name}\nRecall: {m['recall']:.1%} | Precision: {m['precision']:.1%}",
            weight="bold",
            fontsize=11,
        )
        axes[idx].set_xlabel("Predicted Label (0: Legit, 1: Fraud)")
        axes[idx].set_ylabel("Actual Label (0: Legit, 1: Fraud)")
        axes[idx].set_xticklabels(["Genuine", "Fraud"])
        axes[idx].set_yticklabels(["Genuine", "Fraud"])
        
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")
        
    plt.suptitle("Confusion Matrices: Catching Frauds (TP) vs False Alarms (FP)", fontsize=15, weight="bold")
    plt.tight_layout()
    output_path = FIGURES_DIR / "07_confusion_matrices_grid.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_financial_cost_benefit(metrics, avg_fraud_loss: float = 122.21, cost_per_false_alarm: float = 15.0):
    """
    Simulate financial loss under each model.
    
    - Cost of False Negative (FN): Stolen amount unrecovered (avg ~$122.21 from EDA)
    - Cost of False Positive (FP): Operational verification cost + customer friction (~$15.00)
    """
    costs = {}
    for name, m in metrics.items():
        fn_cost = m["fn"] * avg_fraud_loss
        fp_cost = m["fp"] * cost_per_false_alarm
        total_cost = fn_cost + fp_cost
        costs[name] = {
            "Missed Fraud Cost (FN)": fn_cost,
            "False Alarm Cost (FP)": fp_cost,
            "Total Financial Loss": total_cost,
        }
        
    cost_df = pd.DataFrame(costs).T.sort_values(by="Total Financial Loss")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cost_df[["Missed Fraud Cost (FN)", "False Alarm Cost (FP)"]].plot(
        kind="bar",
        stacked=True,
        color=["#d9534f", "#f0ad4e"],
        edgecolor="black",
        ax=ax,
    )
    plt.title(
        f"Financial Loss Analysis (FN Loss @ ${avg_fraud_loss:.2f} + FP Friction @ ${cost_per_false_alarm:.2f})",
        weight="bold",
        fontsize=13,
    )
    plt.ylabel("Estimated Financial Cost ($)", fontsize=11)
    plt.xlabel("Model Configuration", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    
    for idx, (model_name, row) in enumerate(cost_df.iterrows()):
        total = row["Total Financial Loss"]
        ax.annotate(
            f"${total:,.0f}",
            xy=(idx, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
        
    plt.tight_layout()
    output_path = FIGURES_DIR / "08_financial_cost_analysis.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def generate_markdown_summary(metrics) -> str:
    """Generate Markdown comparison table for README."""
    lines = [
        "| Model Architecture | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | False Positives (FP) | Missed Frauds (FN) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, m in sorted(metrics.items(), key=lambda x: x[1]["pr_auc"], reverse=True):
        lines.append(
            f"| **{name}** | **{m['pr_auc']:.4f}** | {m['roc_auc']:.4f} | {m['recall']:.2%} | {m['precision']:.2%} | {m['f1']:.4f} | {m['fp']:,} | {m['fn']:,} |"
        )
    table_str = "\n".join(lines)
    
    with open(REPORTS_DIR / "metrics_table.md", "w") as f:
        f.write(table_str)
        
    return table_str


def run_evaluation():
    """Execute complete evaluation visualization pipeline."""
    metrics, curves = load_results()
    plot_pr_and_roc_curves(metrics, curves)
    plot_metrics_comparison_bar(metrics)
    plot_confusion_matrices(metrics)
    plot_financial_cost_benefit(metrics)
    summary_table = generate_markdown_summary(metrics)
    print("\n--- MODEL PERFORMANCE SUMMARY TABLE ---\n")
    print(summary_table)


if __name__ == "__main__":
    run_evaluation()
