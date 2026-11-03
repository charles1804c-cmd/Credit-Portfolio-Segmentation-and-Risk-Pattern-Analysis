"""
Phase 1 — Data Collection, Cleaning, and Preprocessing.
Loads the Lending Club dataset, inspects variables, removes unnecessary identifiers,
handles missing values, parses dates, maps employment length, and saves cleaned data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = Path("/Users/charles/Downloads/loans_granting_model_dataset.csv")
CLEANED_PATH = BASE_DIR / "data" / "processed" / "loans_cleaned.csv"
SAMPLE_PATH = BASE_DIR / "data" / "processed" / "loans_cleaned_sample.csv"


def clean_employment_length(val):
    """Convert employment length string to numeric years."""
    if pd.isna(val) or val == "NI":
        return np.nan
    val_str = str(val).strip()
    if "< 1 year" in val_str or "<1 year" in val_str:
        return 0.5
    if "10+" in val_str:
        return 10.0
    digits = [c for c in val_str if c.isdigit()]
    if digits:
        return float("".join(digits))
    return np.nan


def run_data_cleaning():
    print(f"Loading raw dataset from {RAW_PATH}...")
    df = pd.read_csv(
        RAW_PATH,
        low_memory=False,
        dtype={
            "id": int,
            "loan_amnt": float,
            "revenue": float,
            "dti_n": float,
            "fico_n": float,
            "Default": int,
        },
    )
    initial_rows = len(df)
    print(f"Initial raw rows: {initial_rows:,}, columns: {len(df.columns)}")

    # 1. Drop exact duplicates if any
    df = df.drop_duplicates()
    print(f"Rows after dropping exact duplicates: {len(df):,}")

    # 2. Exclude text columns and uninformative constants
    # 'title' and 'desc' are free-text; 'experience_c' is 99.998% 1; 'id' is identifier
    # 'zip_code' has high cardinality and 'addr_state' represents geography properly
    cols_to_drop = [c for c in ["title", "desc", "experience_c"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Clean and parse employment length
    df["emp_length_years"] = df["emp_length"].apply(clean_employment_length)
    emp_median = df["emp_length_years"].median()
    df["emp_length_years"] = df["emp_length_years"].fillna(emp_median)

    # 4. Clean date column 'issue_d'
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_date"].dt.year
    df["issue_month"] = df["issue_date"].dt.month

    # 5. Handle financial outliers and anomalies:
    # dti_n max is 999 which is anomalous/data entry error in lending club
    # Annual revenue has extreme outliers (e.g., $10M+)
    # Filter realistic ranges: dti between 0 and 60 (Lending Club cap is 40-50 usually)
    # Revenue > 0 and <= 500,000 for standard retail portfolio
    df["dti_clean"] = df["dti_n"].clip(lower=0, upper=60)
    df["revenue_clean"] = df["revenue"].clip(lower=5000, upper=300000)

    # 6. Simplify home ownership: consolidate rare 'OTHER' into 'RENT' or 'OTHER'
    df["home_ownership_clean"] = df["home_ownership_n"].replace({"NONE": "OTHER", "ANY": "OTHER"})

    # 7. Group rare purposes: keep top 8, rest to 'other'
    top_purposes = df["purpose"].value_counts().nlargest(8).index.tolist()
    df["purpose_clean"] = df["purpose"].apply(lambda x: x if x in top_purposes else "other")

    # 8. Derived financial ratios
    # Loan to Income Ratio
    df["loan_to_income"] = (df["loan_amnt"] / df["revenue_clean"]).clip(0, 1.5)
    # Estimated monthly debt obligation = (dti / 100) * (revenue / 12)
    df["monthly_debt_est"] = (df["dti_clean"] / 100.0) * (df["revenue_clean"] / 12.0)

    print("Data cleaning complete.")
    print("Summary of Key Features:")
    print(df[["loan_amnt", "revenue_clean", "dti_clean", "fico_n", "emp_length_years", "Default"]].describe().round(2))

    # Save a stratified representative sample for heavy visual algorithms (UMAP, t-SNE, Hierarchical, DBSCAN)
    # 50,000 rows sample stratified by Default status to preserve exact population default rate
    print("\nCreating stratified sample of 50,000 rows for high-fidelity clustering & visual embeddings...")
    sample_df = df.groupby("Default", group_keys=False).apply(
        lambda x: x.sample(n=int(50000 * len(x) / len(df)), random_state=42)
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)

    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving sample dataset to {SAMPLE_PATH}...")
    sample_df.to_csv(SAMPLE_PATH, index=False)
    
    # Save a lighter overview dataset (100k) for full speed and repo convenience
    print(f"Sample default rate: {sample_df['Default'].mean():.4f} (Population: {df['Default'].mean():.4f})")
    
    return sample_df


if __name__ == "__main__":
    run_data_cleaning()
