"""
Phase 4 & 5 — Feature Engineering, Scaling, and Dimensionality Reduction.
Encodes categoricals, applies RobustScaler/StandardScaler, fits PCA, t-SNE, and UMAP,
and saves the transformed feature matrices and pipeline artifacts.
NOTE: Excludes 'Default' and identifiers from clustering inputs (strictly as mandated).
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "loans_cleaned_sample.csv"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
TBL_DIR = BASE_DIR / "reports" / "tables"


def run_feature_engineering_and_dim_reduction():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data for feature engineering...")
    df = pd.read_csv(DATA_PATH)

    # Core clustering features (Strictly financial, credit, employment, and loan characteristics)
    num_features = ["loan_amnt", "revenue_clean", "dti_clean", "fico_n", "emp_length_years", "loan_to_income"]
    cat_features = ["home_ownership_clean", "purpose_clean"]

    # 1. Scaling numerical variables
    # We use RobustScaler on financial amounts (loan, revenue, dti) and StandardScaler on credit score & ratios
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_features])
    X_num_df = pd.DataFrame(X_num, columns=num_features)

    # 2. One-hot encoding categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    X_cat = encoder.fit_transform(df[cat_features])
    cat_cols = encoder.get_feature_names_out(cat_features).tolist()
    X_cat_df = pd.DataFrame(X_cat, columns=cat_cols)

    # Combine into full clustering feature matrix
    X_all = pd.concat([X_num_df, X_cat_df], axis=1)
    feature_names = X_all.columns.tolist()
    print(f"Total features for clustering: {len(feature_names)} features.")

    # 3. PCA Dimensionality Reduction
    print("Fitting PCA...")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_all)
    exp_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Save PCA components table
    pca_var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(exp_var))],
        "Explained_Variance_Ratio": pca_full.explained_variance_ratio_,
        "Cumulative_Explained_Variance": exp_var
    })
    pca_var_df.to_csv(TBL_DIR / "pca_variance_explained.csv", index=False)

    # Fit 2D and 3D PCA for visualization & fast clustering
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_all)

    pca_5d = PCA(n_components=5, random_state=42)
    X_pca_5d = pca_5d.fit_transform(X_all)
    var_5d = pca_5d.explained_variance_ratio_.sum()
    print(f"5-component PCA retains {var_5d*100:.2f}% of total variance.")

    # 4. t-SNE Embedding (on a high-density 5,000 sub-sample for crisp visualization)
    print("Computing t-SNE (5,000 sample)...")
    sub_idx = np.random.RandomState(42).choice(len(X_all), size=5000, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=35, max_iter=1000)
    X_tsne_sub = tsne.fit_transform(X_pca_5d[sub_idx])

    # 5. UMAP Embedding (5,000 sample for lightning fast & topology preserving projection)
    print("Computing UMAP (5,000 sample)...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    X_umap_sub = umap_reducer.fit_transform(X_pca_5d[sub_idx])

    # Save preprocessed matrices and artifacts
    artifacts = {
        "scaler": scaler,
        "encoder": encoder,
        "num_features": num_features,
        "cat_features": cat_features,
        "feature_names": feature_names,
        "pca_2d": pca_2d,
        "pca_5d": pca_5d,
        "X_pca_5d": X_pca_5d,
        "X_pca_2d": X_pca_2d,
        "sub_idx": sub_idx,
        "X_tsne_sub": X_tsne_sub,
        "X_umap_sub": X_umap_sub,
    }

    with open(MODEL_DIR / "feature_pipeline_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    # Save numerical matrices for clustering
    np.save(MODEL_DIR / "X_scaled.npy", X_all.values)
    np.save(MODEL_DIR / "X_pca_5d.npy", X_pca_5d)
    np.save(MODEL_DIR / "X_pca_2d.npy", X_pca_2d)

    print("Saved feature pipeline artifacts and matrices.")
    return artifacts


if __name__ == "__main__":
    run_feature_engineering_and_dim_reduction()
