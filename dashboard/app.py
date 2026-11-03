"""
Interactive Streamlit Dashboard: Credit Portfolio Segmentation & Risk Pattern Analysis.
Provides executive-level loan portfolio analytics across 6 modules:
1. Executive Portfolio Overview
2. Exploratory Data Analysis & Risk Drivers
3. Dimensionality Reduction (PCA, t-SNE, UMAP)
4. Clustering Comparison & Borrower Segments
5. Historical Default Pattern Analysis
6. Real-time Borrower Scoring & Segment Predictor
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Credit Portfolio Segmentation & Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "loans_segmented.csv"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
TBL_DIR = BASE_DIR / "reports" / "tables"

# ----------------- DATA & MODEL CACHING -----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    cluster_names = {
        0: "Cluster 0: Moderate-Income Mortgagors (Low-Risk)",
        1: "Cluster 1: Entry/Budget Renters (Moderate-Risk)",
        2: "Cluster 2: Overleveraged High-DTI Borrowers (High-Risk)",
        3: "Cluster 3: Affluent High-Income Borrowers (Prime Low-Risk)"
    }
    df["Cluster_Name"] = df["kmeans_cluster"].map(cluster_names)
    df["Default_Status"] = df["Default"].map({0: "Fully Paid / Current", 1: "Default / Charged-Off"})
    return df

@st.cache_resource
def load_models_and_artifacts():
    with open(MODEL_DIR / "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(MODEL_DIR / "feature_pipeline_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return kmeans, artifacts

df_raw = load_data()
kmeans_model, artifacts = load_models_and_artifacts()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
st.sidebar.title("Portfolio Filter Center")

# Filter by Home Ownership
all_homes = sorted(df_raw["home_ownership_clean"].unique().tolist())
sel_homes = st.sidebar.multiselect("Home Ownership", all_homes, default=all_homes)

# Filter by Loan Purpose
all_purposes = sorted(df_raw["purpose_clean"].unique().tolist())
sel_purposes = st.sidebar.multiselect("Loan Purpose", all_purposes, default=all_purposes)

# Filter by FICO Range
min_fico, max_fico = int(df_raw["fico_n"].min()), int(df_raw["fico_n"].max())
sel_fico = st.sidebar.slider("FICO Score Range", min_fico, max_fico, (min_fico, max_fico))

# Filter by Loan Amount
min_loan, max_loan = int(df_raw["loan_amnt"].min()), int(df_raw["loan_amnt"].max())
sel_loan = st.sidebar.slider("Loan Amount ($)", min_loan, max_loan, (min_loan, max_loan), step=1000)

# Filter DataFrame
filtered_df = df_raw[
    (df_raw["home_ownership_clean"].isin(sel_homes)) &
    (df_raw["purpose_clean"].isin(sel_purposes)) &
    (df_raw["fico_n"] >= sel_fico[0]) & (df_raw["fico_n"] <= sel_fico[1]) &
    (df_raw["loan_amnt"] >= sel_loan[0]) & (df_raw["loan_amnt"] <= sel_loan[1])
]

# ----------------- MAIN HEADER -----------------
st.markdown("<h1 style='color:#1E3A8A;'>🏦 Credit Portfolio Segmentation & Risk Pattern Analysis</h1>", unsafe_allow_html=True)
st.markdown("##### *Unsupervised Learning, Manifold Visualizations, and Default Pattern Discovery for Banking & Credit Portfolios*")
st.markdown("---")

# ----------------- TABS -----------------
tabs = st.tabs([
    "📊 Executive Overview",
    "📈 Exploratory Analysis (EDA)",
    "🌌 Manifold Embeddings (PCA/t-SNE/UMAP)",
    "🧩 Borrower Segments & Algorithms",
    "⚠️ Default Risk Patterns",
    "🎯 Live Borrower Segment Predictor"
])

# ----------------- TAB 1: EXECUTIVE OVERVIEW -----------------
with tabs[0]:
    st.subheader("Key Portfolio Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loans in View", f"{len(filtered_df):,}", delta=f"{len(filtered_df)/len(df_raw)*100:.1f}% of cohort")
    with col2:
        st.metric("Total Funded Capital", f"${filtered_df['loan_amnt'].sum()/1e6:.2f} M")
    with col3:
        avg_fico = filtered_df["fico_n"].mean()
        st.metric("Portfolio Average FICO", f"{avg_fico:.1f}", delta=f"{avg_fico - df_raw['fico_n'].mean():+.1f}")
    with col4:
        def_rate = filtered_df["Default"].mean() * 100
        st.metric("Portfolio Default Rate", f"{def_rate:.2f}%", delta=f"{def_rate - df_raw['Default'].mean()*100:+.2f}%", delta_color="inverse")

    st.markdown("### Portfolio Distribution by Segment Archetype")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        seg_summary = filtered_df.groupby("Cluster_Name").agg(
            Borrower_Count=("id", "count"),
            Avg_Loan=("loan_amnt", "mean"),
            Avg_Income=("revenue_clean", "mean"),
            Avg_DTI=("dti_clean", "mean"),
            Default_Rate=("Default", lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        fig_bar = px.bar(
            seg_summary,
            x="Cluster_Name",
            y="Borrower_Count",
            color="Default_Rate",
            color_continuous_scale="Reds",
            title="Borrowers per Cluster (Color = Historical Default Rate %)",
            labels={"Borrower_Count": "Borrowers", "Cluster_Name": "Segment", "Default_Rate": "Default %"}
        )
        fig_bar.update_layout(xaxis_tickangle=-15, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_pie = px.pie(
            filtered_df,
            names="Cluster_Name",
            title="Portfolio Composition Share",
            hole=0.45,
            color_discrete_sequence=["#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c"]
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- TAB 2: EXPLORATORY ANALYSIS -----------------
with tabs[1]:
    st.subheader("Credit & Financial Feature Distributions")
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        fig_fico = px.histogram(filtered_df, x="fico_n", nbins=35, color="Default_Status",
                                title="FICO Score Distribution by Default Status",
                                color_discrete_map={"Fully Paid / Current": "#2a9d8f", "Default / Charged-Off": "#e63946"},
                                barmode="overlay", opacity=0.7)
        st.plotly_chart(fig_fico, use_container_width=True)
        
        fig_box = px.box(filtered_df, x="home_ownership_clean", y="loan_amnt", color="Default_Status",
                         title="Loan Amount by Home Ownership & Default Status",
                         color_discrete_map={"Fully Paid / Current": "#2a9d8f", "Default / Charged-Off": "#e63946"})
        st.plotly_chart(fig_box, use_container_width=True)

    with eda_col2:
        fig_dti = px.histogram(filtered_df, x="dti_clean", nbins=35, color="Default_Status",
                               title="Debt-to-Income (DTI %) Distribution",
                               color_discrete_map={"Fully Paid / Current": "#2a9d8f", "Default / Charged-Off": "#e63946"},
                               barmode="overlay", opacity=0.7)
        st.plotly_chart(fig_dti, use_container_width=True)

        fig_purp = px.histogram(filtered_df, y="purpose_clean", color="Default_Status",
                                title="Loan Purpose Volume and Risk Breakdown",
                                color_discrete_map={"Fully Paid / Current": "#2a9d8f", "Default / Charged-Off": "#e63946"},
                                orientation="h")
        st.plotly_chart(fig_purp, use_container_width=True)

# ----------------- TAB 3: MANIFOLD EMBEDDINGS -----------------
with tabs[2]:
    st.subheader("Dimensionality Reduction & Manifold Exploration")
    st.markdown("""
    High-dimensional financial profiles (loan amount, income, DTI, FICO, employment length, loan-to-income, and categorical dummies) 
    projected into 2D continuous spaces using **PCA**, **t-SNE**, and **UMAP**.
    """)
    
    sub_idx = artifacts["sub_idx"]
    sub_df = df_raw.iloc[sub_idx].copy().reset_index(drop=True)
    
    vis_choice = st.radio("Color Embedding Points By:", ["Borrower Cluster Segment", "Default Outcome", "DBSCAN Anomalies"], horizontal=True)

    dim_c1, dim_c2, dim_c3 = st.columns(3)
    
    if vis_choice == "Borrower Cluster Segment":
        color_col = sub_df["Cluster_Name"]
        color_map = None
    elif vis_choice == "Default Outcome":
        color_col = sub_df["Default_Status"]
        color_map = {"Fully Paid / Current": "#2a9d8f", "Default / Charged-Off": "#e63946"}
    else:
        color_col = (sub_df["dbscan_cluster"] == -1).map({True: "Anomaly / Outlier", False: "Core Portfolio"})
        color_map = {"Core Portfolio": "#b0c4de", "Anomaly / Outlier": "#e63946"}

    with dim_c1:
        st.markdown("**1. PCA (Principal Component Analysis)**")
        st.caption("Captures 85.9% variance across top components; linear orthogonal projection.")
        fig_pca = px.scatter(x=artifacts["X_pca_2d"][sub_idx, 0], y=artifacts["X_pca_2d"][sub_idx, 1],
                             color=color_col, color_discrete_map=color_map, opacity=0.6,
                             labels={"x": "PC 1", "y": "PC 2"})
        fig_pca.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_pca, use_container_width=True)

    with dim_c2:
        st.markdown("**2. t-SNE (Local Topology Embedding)**")
        st.caption("Preserves local neighborhood distances; highlights dense non-linear clusters.")
        fig_tsne = px.scatter(x=artifacts["X_tsne_sub"][:, 0], y=artifacts["X_tsne_sub"][:, 1],
                              color=color_col, color_discrete_map=color_map, opacity=0.6,
                              labels={"x": "t-SNE 1", "y": "t-SNE 2"})
        fig_tsne.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_tsne, use_container_width=True)

    with dim_c3:
        st.markdown("**3. UMAP (Uniform Manifold Projection)**")
        st.caption("Maintains both global hierarchy and local distance metrics efficiently.")
        fig_umap = px.scatter(x=artifacts["X_umap_sub"][:, 0], y=artifacts["X_umap_sub"][:, 1],
                              color=color_col, color_discrete_map=color_map, opacity=0.6,
                              labels={"x": "UMAP 1", "y": "UMAP 2"})
        fig_umap.update_layout(height=420)
        st.plotly_chart(fig_umap, use_container_width=True)

# ----------------- TAB 4: SEGMENTS & ALGORITHMS -----------------
with tabs[3]:
    st.subheader("Clustering Algorithm Benchmarking & Segment Profiles")
    
    # Load comparison table
    comp_df = pd.read_csv(TBL_DIR / "clustering_algorithm_comparison.csv")
    st.markdown("### Algorithmic Comparison: K-Means vs. Hierarchical vs. DBSCAN")
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Segment Financial Profiles & Characteristics")
    prof_df = pd.read_csv(TBL_DIR / "borrower_segment_profiles.csv")
    st.dataframe(prof_df, use_container_width=True)

    col_a, col_b = col_b, col_a = st.columns(2)
    with col_a:
        st.markdown("#### Segment Radar / Multi-Attribute Comparison")
        radar_categories = ["Mean FICO", "Mean DTI", "Mean Loan ($1k)", "Mean Income ($10k)", "Default Rate %"]
        
        fig_radar = go.Figure()
        for idx, row in prof_df.iterrows():
            values = [
                row["Mean_FICO"] / 8.5,
                row["Mean_DTI"] * 2.5,
                row["Mean_Loan_Amount"] / 300,
                row["Mean_Income"] / 1500,
                row["Default_Rate_Pct"] * 2.5
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_categories,
                fill='toself',
                name=f"C{row['kmeans_cluster']}: {row['Cluster_Name'].split(':')[1].split('(')[0].strip()}"
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        st.markdown("#### Anomaly Detection Insights (DBSCAN Outliers)")
        anom_df = pd.read_csv(TBL_DIR / "anomaly_detection_summary.csv")
        st.dataframe(anom_df, use_container_width=True)
        st.info("💡 **Anomaly Characteristics:** DBSCAN isolated extreme-profile borrowers with high incomes ($151k avg), high debt burdens (DTI 29.8%), and maxed-out loans ($24.5k avg). These require targeted manual underwriting.")

# ----------------- TAB 5: DEFAULT RISK PATTERNS -----------------
with tabs[4]:
    st.subheader("Default Risk Pattern Analysis (Post-Clustering Ground Truth Validation)")
    st.markdown("""
    > **Methodological Note:** As mandated by project guidelines, the `Default` status was **strictly excluded** from clustering features. 
    > Below is the post-hoc validation proving that unsupervised grouping effectively isolated real credit default risk.
    """)

    def_c1, def_c2 = st.columns([1.2, 1])
    with def_c1:
        prof_df = pd.read_csv(TBL_DIR / "borrower_segment_profiles.csv")
        fig_def_bar = px.bar(
            prof_df,
            x="Cluster_Name",
            y="Default_Rate_Pct",
            color="Default_Rate_Pct",
            color_continuous_scale="Reds",
            title="Historical Default Rate (%) by Unsupervised Cluster",
            text="Default_Rate_Pct"
        )
        overall_d = df_raw["Default"].mean() * 100
        fig_def_bar.add_hline(y=overall_d, line_dash="dash", line_color="black",
                              annotation_text=f"Portfolio Baseline ({overall_d:.2f}%)")
        fig_def_bar.update_layout(xaxis_tickangle=-15, height=420)
        st.plotly_chart(fig_def_bar, use_container_width=True)

    with def_c2:
        st.markdown("### Risk Tier Evaluation")
        st.error("**Highest Risk:** Cluster 2 (Overleveraged High-DTI Borrowers) — **27.87% Default Rate** (1.40x higher than portfolio baseline). High loan-to-income ratio and heavy consolidation debt.")
        st.warning("**Moderate Risk:** Cluster 1 (Entry/Budget Renters) — **18.16% Default Rate**. Lower income and rental status, but smaller loan sizes constrain risk.")
        st.info("**Low Risk:** Cluster 0 (Moderate-Income Mortgagors) — **16.54% Default Rate**. Stable mortgage equity and balanced leverage.")
        st.success("**Prime Low Risk:** Cluster 3 (Affluent Borrowers) — **15.21% Default Rate** (Lowest in entire portfolio). High FICO (714 avg) and substantial income buffer.")

# ----------------- TAB 6: LIVE PREDICTOR -----------------
with tabs[5]:
    st.subheader("🎯 Live Loan Applicant Segmentation & Risk Scoring")
    st.markdown("Enter applicant attributes to instantly assign their borrower cluster, expected default risk tier, and underwriting recommendations.")

    with st.form("applicant_form"):
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            in_loan = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=1000)
            in_income = st.number_input("Annual Revenue / Income ($)", min_value=10000, max_value=300000, value=75000, step=5000)
        with p_c2:
            in_fico = st.number_input("FICO Credit Score", min_value=600, max_value=850, value=700, step=5)
            in_dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=60.0, value=18.5, step=0.5)
        with p_c3:
            in_emp = st.slider("Employment Length (Years)", 0.5, 10.0, 5.0, step=0.5)
            in_home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
            in_purpose = st.selectbox("Loan Purpose", all_purposes)
        
        submitted = st.form_submit_button("Classify Borrower Segment & Assess Risk", use_container_width=True)

    if submitted:
        # Preprocess single applicant
        scaler = artifacts["scaler"]
        encoder = artifacts["encoder"]
        num_features = artifacts["num_features"]
        cat_features = artifacts["cat_features"]

        loan_to_income = min(in_loan / in_income, 1.5)
        raw_num = pd.DataFrame([[in_loan, in_income, in_dti, in_fico, in_emp, loan_to_income]], columns=num_features)
        raw_cat = pd.DataFrame([[in_home, in_purpose]], columns=cat_features)

        scaled_num = scaler.transform(raw_num)
        encoded_cat = encoder.transform(raw_cat)
        applicant_vec = np.hstack([scaled_num, encoded_cat])

        pred_cluster = kmeans_model.predict(applicant_vec)[0]

        cluster_descriptions = {
            0: ("Cluster 0: Moderate-Income Mortgagors", "Low Risk", "16.54%", "#2b5c8f", "Approve standard rates. Low default risk with dependable home equity backing."),
            1: ("Cluster 1: Entry/Budget Renters", "Moderate Risk", "18.16%", "#e9c46a", "Approve with standard interest margins. Recommend automated monthly repayment setup."),
            2: ("Cluster 2: Overleveraged High-DTI Borrowers", "High Risk", "27.87%", "#e63946", "Require secondary review or co-signer. High default likelihood due to debt stress."),
            3: ("Cluster 3: Affluent High-Income Borrowers", "Prime Low Risk", "15.21%", "#2a9d8f", "Fast-track approval. Eligible for prime interest rate discounts and larger credit lines.")
        }

        c_name, c_risk, c_def, c_color, c_rec = cluster_descriptions[pred_cluster]

        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 6px solid {c_color}; padding: 20px; border-radius: 6px; margin-top: 20px;">
            <h3 style="color: {c_color}; margin-top: 0;">Assigned Segment: {c_name}</h3>
            <p><strong>Predicted Risk Tier:</strong> {c_risk} | <strong>Historical Cohort Default Rate:</strong> {c_def}</p>
            <p><strong>Underwriting Recommendation:</strong> {c_rec}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Credit Portfolio Segmentation & Risk Pattern Analysis | GUVI Capstone Project</div>", unsafe_allow_html=True)
