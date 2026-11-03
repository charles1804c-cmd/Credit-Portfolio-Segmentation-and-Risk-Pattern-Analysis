# Credit Portfolio Segmentation and Risk Pattern Analysis

An unsupervised machine learning and loan portfolio risk profiling project designed for banking and retail credit management. This system groups loan applicants based on financial, credit, and employment characteristics, discovers latent risk archetypes, validates them against historical default outcomes, isolates portfolio anomalies, and deploys an interactive Streamlit analytics dashboard.

---

## 📌 Project Overview & Problem Statement
Financial institutions manage diverse borrower portfolios with varying income profiles, credit histories, debt obligations, and loan requirements. 

This project implements an end-to-end unsupervised learning framework using the **Lending Club** loan granting dataset (>1.34 million records):
1. **Discovers natural borrower segments** without using credit outcome labels during training.
2. **Projects multi-dimensional financial features** into 2D continuous space via **PCA**, **t-SNE**, and **UMAP**.
3. **Benchmarks three clustering paradigms**: **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN**.
4. **Validates cluster quality** by comparing historical default rates across the discovered unsupervised segments.
5. **Detects high-risk portfolio anomalies** using density-based outlier analysis.
6. **Delivers an interactive Streamlit dashboard** with live borrower risk scoring.

---

## 🚀 Quick Setup & Execution

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/credit-portfolio-segmentation.git
cd "credit-portfolio-segmentation"
pip install -r requirements.txt
```

### 2. Reproduce the Full Pipeline
Execute the modular scripts in sequence:
```bash
# Phase 1: Data Cleaning & Preprocessing
python src/data_cleaning.py

# Phase 2: Exploratory Data Analysis & Statistical Visualizations
python src/eda.py

# Phase 3 & 4: Feature Scaling & Dimensionality Reduction (PCA, t-SNE, UMAP)
python src/feature_engineering.py

# Phase 5: Clustering Training & Comparison (K-Means, Hierarchical, DBSCAN)
python src/train_clustering.py

# Phase 6: Manifold Visualizations
python src/visualize_embeddings.py

# Phase 7: Risk Profiling & Default Pattern Analysis
python src/risk_profiling.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Discovered Borrower Segments & Default Risk Profiles

The optimal **K-Means (k=4)** model identified four financially distinct borrower archetypes:

| Segment Archetype | Portfolio Share | Avg Loan ($) | Avg Income ($) | Avg FICO | Avg DTI (%) | Historical Default Rate (%) | Risk Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 3: Affluent Borrowers** | 13.06% | $24,787 | $157,801 | 713.6 | 14.01% | **15.21%** | 🟢 Prime Low-Risk |
| **Cluster 0: Moderate Mortgagors** | 28.76% | $9,281 | $70,711 | 696.7 | 17.45% | **16.54%** | 🔵 Low-Risk |
| **Cluster 1: Entry/Budget Renters**| 30.68% | $9,071 | $59,140 | 693.7 | 16.51% | **18.16%** | 🟡 Moderate-Risk |
| **Cluster 2: Overleveraged High-DTI**| 27.50% | $20,697 | $61,602 | 697.5 | 22.77% | **27.87%** | 🔴 High-Risk |

> **Key Ground-Truth Insight:** Even though the `Default` column was **strictly excluded** from clustering, the unsupervised model cleanly separated borrowers with a **15.21% default rate** (Cluster 3) from those with a **27.87% default rate** (Cluster 2) — an **83% relative risk delta**.

---

## 🔬 Clustering Algorithm Comparison

| Algorithm | Number of Clusters | Silhouette Score | Calinski-Harabasz Score | Davies-Bouldin Index | Key Practical Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means** | **4** | **0.1462** | **839.2** | **1.830** | Best scalability, centroid interpretability, instant live applicant scoring. |
| **Hierarchical (Ward)** | 4 | 0.1402 | 425.1 | 1.839 | Deterministic hierarchical taxonomy, dendrogram branch traceability. |
| **DBSCAN** | 1 (Core) + Noise | 0.0812 | 124.5 | 1.940 | Arbitrary shape grouping, automated high-risk anomaly isolation (370 outliers). |

---

## 🗂️ Project Directory Structure

```
Credit Portfolio Segmentation and Risk Pattern Analysis/
│
├── data/
│   ├── raw/                                 # Raw dataset folder
│   └── processed/                           # Cleaned & segmented data CSVs
│       ├── loans_cleaned_sample.csv
│       └── loans_segmented.csv
│
├── notebooks/                               # Guided analysis Jupyter notebooks
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_dimensionality_reduction.ipynb
│   └── 03_clustering_and_risk_analysis.ipynb
│
├── src/                                     # Modular Python source code
│   ├── data_cleaning.py                     # Missing value handling & cleaning
│   ├── eda.py                               # Statistical visualization generator
│   ├── feature_engineering.py               # Scaling & PCA/t-SNE/UMAP fittings
│   ├── train_clustering.py                  # K-Means, Hierarchical & DBSCAN
│   ├── visualize_embeddings.py              # Publication-grade manifold plots
│   └── risk_profiling.py                    # Cluster risk profiling & defaults
│
├── models/                                  # Serialized pipelines and models
│   ├── kmeans_model.pkl
│   ├── dbscan_model.pkl
│   ├── feature_pipeline_artifacts.pkl
│   ├── X_scaled.npy
│   ├── X_pca_5d.npy
│   └── X_pca_2d.npy
│
├── reports/                                 # Output artifacts & figures
│   ├── figures/                             # Generated PNG visualization plots
│   │   ├── 01_feature_distributions.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_default_by_fico_tier.png
│   │   ├── 04_default_by_category.png
│   │   ├── 05_kmeans_elbow_silhouette.png
│   │   ├── 06_hierarchical_dendrogram.png
│   │   ├── 07_dimensionality_reduction_clusters.png
│   │   ├── 08_dimensionality_reduction_defaults.png
│   │   ├── 09_dbscan_anomalies_umap.png
│   │   ├── 10_cluster_default_rates.png
│   │   └── 11_cluster_financial_comparison.png
│   └── tables/                              # Exported CSV tables & statistics
│       ├── summary_statistics.csv
│       ├── pca_variance_explained.csv
│       ├── clustering_algorithm_comparison.csv
│       ├── borrower_segment_profiles.csv
│       └── anomaly_detection_summary.csv
│
├── dashboard/                               # Streamlit Web Application
│   └── app.py
│
├── requirements.txt                         # Dependencies
├── .gitignore                               # Git ignore configuration
└── README.md                                # Project documentation
```

---

## 💼 Business Recommendations
1. **Dynamic Risk-Based Pricing:** Offer promotional interest rate discounts to **Cluster 3 (Affluent)** to expand prime portfolio share; adjust rates upward on **Cluster 2 (High-DTI)** to account for the 27.87% default rate.
2. **Automated Collateral / Co-Signer Rules:** Flag applicants matching **Cluster 2** for mandatory debt consolidation verification or require co-signers when loan-to-income exceeds 0.30.
3. **Specialized Anomaly Auditing:** Automatically route applicants categorized as **DBSCAN Anomalies** to senior credit risk officers for manual underwriting.
