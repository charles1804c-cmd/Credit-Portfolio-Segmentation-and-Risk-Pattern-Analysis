"""
Phase 7, 8 & 9 — Cluster Profiling, Default Pattern Analysis, and Anomaly Investigation.
1. Computes mean & median characteristics for each cluster (Loan amount, income, DTI, FICO, employment).
2. Calculates historical default rate per cluster (unsupervised validation against ground truth).
3. Evaluates anomalies identified by DBSCAN and creates detailed risk profiles.
Saves summary CSV tables and visualization charts.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "loans_segmented.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
TBL_DIR = BASE_DIR / "reports" / "tables"


def run_risk_profiling():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading segmented dataset...")
    df = pd.read_csv(DATA_PATH)

    # Accurate cluster names derived from empirical financial metrics
    cluster_names = {
        0: "Cluster 0: Moderate-Income Mortgagors (Low-Risk)",
        1: "Cluster 1: Entry/Budget Renters (Moderate-Risk)",
        2: "Cluster 2: Overleveraged High-DTI Borrowers (High-Risk)",
        3: "Cluster 3: Affluent High-Income Borrowers (Prime Low-Risk)"
    }
    df["Cluster_Name"] = df["kmeans_cluster"].map(cluster_names)

    # 1. Cluster Profiling Table
    profile_df = df.groupby(["kmeans_cluster", "Cluster_Name"]).agg(
        Borrower_Count=("id", "count"),
        Portfolio_Share=("id", lambda x: round(len(x) / len(df) * 100, 2)),
        Mean_Loan_Amount=("loan_amnt", "mean"),
        Median_Loan_Amount=("loan_amnt", "median"),
        Mean_Income=("revenue_clean", "mean"),
        Median_Income=("revenue_clean", "median"),
        Mean_DTI=("dti_clean", "mean"),
        Mean_FICO=("fico_n", "mean"),
        Mean_Emp_Years=("emp_length_years", "mean"),
        Mean_Loan_To_Income=("loan_to_income", "mean"),
        Default_Rate_Pct=("Default", lambda x: round(x.mean() * 100, 2))
    ).reset_index()

    # Format numbers
    profile_df["Mean_Loan_Amount"] = profile_df["Mean_Loan_Amount"].round(2)
    profile_df["Median_Loan_Amount"] = profile_df["Median_Loan_Amount"].round(2)
    profile_df["Mean_Income"] = profile_df["Mean_Income"].round(2)
    profile_df["Median_Income"] = profile_df["Median_Income"].round(2)
    profile_df["Mean_DTI"] = profile_df["Mean_DTI"].round(2)
    profile_df["Mean_FICO"] = profile_df["Mean_FICO"].round(1)
    profile_df["Mean_Emp_Years"] = profile_df["Mean_Emp_Years"].round(1)
    profile_df["Mean_Loan_To_Income"] = profile_df["Mean_Loan_To_Income"].round(3)

    profile_df.to_csv(TBL_DIR / "borrower_segment_profiles.csv", index=False)
    print("\n--- Borrower Segment Profiles ---")
    print(profile_df[["Cluster_Name", "Portfolio_Share", "Mean_Loan_Amount", "Mean_Income", "Mean_FICO", "Mean_DTI", "Default_Rate_Pct"]].to_string())

    # 2. Historical Default Rate Across Clusters Chart
    plt.figure(figsize=(11, 5.5))
    palette_default_bars = ["#2b5c8f", "#e9c46a", "#e63946", "#2a9d8f"]
    ax = sns.barplot(data=profile_df, x="Cluster_Name", y="Default_Rate_Pct", palette=palette_default_bars)
    plt.title("Historical Default Rate (%) Across Unsupervised Borrower Segments", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Historical Default Rate (%)", fontweight="bold")
    plt.xlabel("Borrower Cluster Archetype", fontweight="bold")
    plt.xticks(rotation=12, ha="right")

    # Add benchmark line for overall portfolio default rate
    overall_default = df["Default"].mean() * 100
    plt.axhline(overall_default, color="black", linestyle="--", alpha=0.7, label=f"Overall Portfolio Benchmark ({overall_default:.2f}%)")
    plt.legend(frameon=True)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 7), textcoords="offset points", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_cluster_default_rates.png", dpi=200)
    plt.close()
    print("Saved 10_cluster_default_rates.png")

    # 3. Anomaly Analysis (DBSCAN Outliers)
    df["Is_Anomaly"] = (df["dbscan_cluster"] == -1)
    anomaly_summary = df.groupby("Is_Anomaly").agg(
        Count=("id", "count"),
        Mean_Loan=("loan_amnt", "mean"),
        Max_Loan=("loan_amnt", "max"),
        Mean_Income=("revenue_clean", "mean"),
        Max_Income=("revenue_clean", "max"),
        Mean_DTI=("dti_clean", "mean"),
        Max_DTI=("dti_clean", "max"),
        Mean_FICO=("fico_n", "mean"),
        Default_Rate=("Default", lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    anomaly_summary["Is_Anomaly"] = anomaly_summary["Is_Anomaly"].map({True: "Anomaly / Noise (DBSCAN -1)", False: "Core Portfolio"})
    anomaly_summary.to_csv(TBL_DIR / "anomaly_detection_summary.csv", index=False)
    print("\n--- Anomaly Detection Analysis ---")
    print(anomaly_summary.round(2).to_string())

    # 4. Multi-Feature Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.barplot(data=profile_df, x="kmeans_cluster", y="Mean_FICO", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Mean FICO Score", fontweight="bold")
    axes[0].set_xlabel("Cluster (0: Mortgagors, 1: Renters, 2: High-DTI, 3: Affluent)")
    axes[0].set_ylim(650, 740)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")

    sns.barplot(data=profile_df, x="kmeans_cluster", y="Mean_DTI", ax=axes[1], palette="Oranges_d")
    axes[1].set_title("Mean Debt-to-Income (DTI %)", fontweight="bold")
    axes[1].set_xlabel("Cluster (0: Mortgagors, 1: Renters, 2: High-DTI, 3: Affluent)")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")

    sns.barplot(data=profile_df, x="kmeans_cluster", y="Mean_Income", ax=axes[2], palette="Greens_d")
    axes[2].set_title("Mean Annual Income ($)", fontweight="bold")
    axes[2].set_xlabel("Cluster (0: Mortgagors, 1: Renters, 2: High-DTI, 3: Affluent)")
    for p in axes[2].patches:
        axes[2].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_cluster_financial_comparison.png", dpi=200)
    plt.close()
    print("Saved 11_cluster_financial_comparison.png")


if __name__ == "__main__":
    run_risk_profiling()
