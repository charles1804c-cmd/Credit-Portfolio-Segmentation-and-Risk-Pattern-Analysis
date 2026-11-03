"""
Phase 2 — Exploratory Data Analysis (EDA).
Generates statistical summaries, correlation heatmaps, feature distributions,
and risk/default pattern visualizations across financial and credit dimensions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "loans_cleaned_sample.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
TBL_DIR = BASE_DIR / "reports" / "tables"


def run_eda():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading sample data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Distribution of Key Numerical Features (FICO, Loan Amount, Revenue, DTI)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["fico_n"], bins=30, kde=True, ax=axes[0, 0], color="#2b5c8f")
    axes[0, 0].set_title("Distribution of FICO Credit Score", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("FICO Score")

    sns.histplot(df["loan_amnt"], bins=30, kde=True, ax=axes[0, 1], color="#2a9d8f")
    axes[0, 1].set_title("Distribution of Loan Amount ($)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Loan Amount ($)")

    sns.histplot(df["revenue_clean"], bins=30, kde=True, ax=axes[1, 0], color="#e76f51")
    axes[1, 0].set_title("Distribution of Annual Revenue / Income ($)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Annual Revenue ($)")

    sns.histplot(df["dti_clean"], bins=30, kde=True, ax=axes[1, 1], color="#e9c46a")
    axes[1, 1].set_title("Distribution of Debt-to-Income Ratio (%)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("DTI (%)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_feature_distributions.png", dpi=200)
    plt.close()
    print("Saved 01_feature_distributions.png")

    # 2. Correlation Matrix
    numeric_cols = ["fico_n", "loan_amnt", "revenue_clean", "dti_clean", "emp_length_years", "loan_to_income", "Default"]
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap: Financial Features & Default", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_correlation_heatmap.png", dpi=200)
    plt.close()
    print("Saved 02_correlation_heatmap.png")

    # 3. Default Rate across FICO Score Buckets
    df["fico_bin"] = pd.cut(df["fico_n"], bins=[600, 660, 700, 740, 780, 850], labels=["600-659 (Poor)", "660-699 (Fair)", "700-739 (Good)", "740-779 (Very Good)", "780+ (Exceptional)"])
    fico_defaults = df.groupby("fico_bin", observed=False)["Default"].agg(["count", "mean"]).reset_index()
    fico_defaults["mean"] *= 100

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=fico_defaults, x="fico_bin", y="mean", palette="Reds_r")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 7), textcoords="offset points", fontweight="bold")
    plt.title("Historical Default Rate (%) by FICO Credit Score Tier", fontsize=13, fontweight="bold")
    plt.ylabel("Default Rate (%)")
    plt.xlabel("FICO Score Tier")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_default_by_fico_tier.png", dpi=200)
    plt.close()
    print("Saved 03_default_by_fico_tier.png")

    # 4. Default Rate by Home Ownership & Purpose
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    home_defaults = df.groupby("home_ownership_clean", observed=False)["Default"].mean().reset_index()
    home_defaults["Default"] *= 100
    sns.barplot(data=home_defaults.sort_values("Default", ascending=False), x="home_ownership_clean", y="Default", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Default Rate by Home Ownership", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].set_xlabel("Home Ownership")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontweight="bold")

    purpose_defaults = df.groupby("purpose_clean", observed=False)["Default"].mean().reset_index()
    purpose_defaults["Default"] *= 100
    sns.barplot(data=purpose_defaults.sort_values("Default", ascending=False), x="Default", y="purpose_clean", ax=axes[1], palette="Oranges_r")
    axes[1].set_title("Default Rate by Loan Purpose", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Default Rate (%)")
    axes[1].set_ylabel("Loan Purpose")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_width():.1f}%", (p.get_width(), p.get_y() + p.get_height() / 2.),
                         ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_default_by_category.png", dpi=200)
    plt.close()
    print("Saved 04_default_by_category.png")

    # Save summary stats table
    summary_stats = df[numeric_cols].describe().T
    summary_stats.to_csv(TBL_DIR / "summary_statistics.csv")
    print("Saved summary_statistics.csv")


if __name__ == "__main__":
    run_eda()
