"""
Phase 5 & 10 — Dimensionality Reduction Visualizations.
Generates publication-quality 2D projections of borrower segments using:
1. PCA (Principal Component Analysis)
2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
3. UMAP (Uniform Manifold Approximation and Projection)
Visualized by Cluster Segment and by Historical Default Status.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "loans_segmented.csv"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"


def run_visualize_embeddings():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data and pipeline artifacts...")
    df = pd.read_csv(DATA_PATH)

    with open(MODEL_DIR / "feature_pipeline_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)

    sub_idx = arts["sub_idx"]
    X_tsne_sub = arts["X_tsne_sub"]
    X_umap_sub = arts["X_umap_sub"]
    X_pca_2d = arts["X_pca_2d"][sub_idx]

    sub_df = df.iloc[sub_idx].copy().reset_index(drop=True)
    sub_df["Cluster_Name"] = sub_df["kmeans_cluster"].map({
        0: "Cluster 0: Moderate-Income Mortgages",
        1: "Cluster 1: Prime High-Income Borrowers",
        2: "Cluster 2: Subprime / High-DTI Renters",
        3: "Cluster 3: Mid-Tier Standard Debt"
    })
    sub_df["Default_Label"] = sub_df["Default"].map({0: "Non-Default (Paid)", 1: "Default / Charged-off"})

    palette_clusters = ["#2b5c8f", "#2a9d8f", "#e76f51", "#e9c46a"]
    palette_default = {"Non-Default (Paid)": "#2a9d8f", "Default / Charged-off": "#e63946"}

    # 1. Tri-Panel Plot: PCA vs t-SNE vs UMAP colored by Cluster Segments
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=sub_df["Cluster_Name"],
                    palette=palette_clusters, alpha=0.6, s=18, ax=axes[0], legend=False)
    axes[0].set_title("PCA Projection (Linear Variance)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")

    sns.scatterplot(x=X_tsne_sub[:, 0], y=X_tsne_sub[:, 1], hue=sub_df["Cluster_Name"],
                    palette=palette_clusters, alpha=0.6, s=18, ax=axes[1], legend=False)
    axes[1].set_title("t-SNE Projection (Local Topology)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    sns.scatterplot(x=X_umap_sub[:, 0], y=X_umap_sub[:, 1], hue=sub_df["Cluster_Name"],
                    palette=palette_clusters, alpha=0.6, s=18, ax=axes[2])
    axes[2].set_title("UMAP Projection (Global & Local Structure)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("UMAP Dimension 1")
    axes[2].set_ylabel("UMAP Dimension 2")
    axes[2].legend(title="Borrower Segments", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_dimensionality_reduction_clusters.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved 07_dimensionality_reduction_clusters.png")

    # 2. Tri-Panel Plot: PCA vs t-SNE vs UMAP colored by Historical Default
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=sub_df["Default_Label"],
                    palette=palette_default, alpha=0.5, s=16, ax=axes[0], legend=False)
    axes[0].set_title("PCA: Default Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")

    sns.scatterplot(x=X_tsne_sub[:, 0], y=X_tsne_sub[:, 1], hue=sub_df["Default_Label"],
                    palette=palette_default, alpha=0.5, s=16, ax=axes[1], legend=False)
    axes[1].set_title("t-SNE: Default Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    sns.scatterplot(x=X_umap_sub[:, 0], y=X_umap_sub[:, 1], hue=sub_df["Default_Label"],
                    palette=palette_default, alpha=0.5, s=16, ax=axes[2])
    axes[2].set_title("UMAP: Default Distribution", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("UMAP Dimension 1")
    axes[2].set_ylabel("UMAP Dimension 2")
    axes[2].legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_dimensionality_reduction_defaults.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved 08_dimensionality_reduction_defaults.png")

    # 3. Anomaly Visualization (DBSCAN Outliers vs Core)
    fig, ax = plt.subplots(figsize=(9, 6))
    sub_df["Anomaly"] = (sub_df["dbscan_cluster"] == -1).map({True: "Anomaly / Outlier", False: "Core Portfolio"})
    sns.scatterplot(x=X_umap_sub[:, 0], y=X_umap_sub[:, 1], hue=sub_df["Anomaly"],
                    palette={"Core Portfolio": "#b0c4de", "Anomaly / Outlier": "#e63946"},
                    style=sub_df["Anomaly"], markers={"Core Portfolio": "o", "Anomaly / Outlier": "X"},
                    s=25, alpha=0.7, ax=ax)
    plt.title("DBSCAN Anomaly Detection on UMAP Manifold", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_dbscan_anomalies_umap.png", dpi=200)
    plt.close()
    print("Saved 09_dbscan_anomalies_umap.png")


if __name__ == "__main__":
    run_visualize_embeddings()
