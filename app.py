# ==================================================
# IMPORTS
# ==================================================
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
import umap.umap_ as umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# CONFIG
# ==================================================
DATA_INDEX_PATH = "data/index.csv"
SAMPLE_ID_COL = "!Sample_geo_accession"
PCR_COLUMN = "!Sample_characteristics_ch1.4"

STUDY_INFO = {
    "GSE41998": {
        "title": (
            "Biomarker analysis of neoadjuvant "
            "Doxorubicin/Cyclophosphamide followed by "
            "Ixabepilone or Paclitaxel in early-stage breast cancer"
        ),
        "abstract": (
            "Predictive biomarkers offer the potential to improve the "
            "benefit:risk ratio of a therapeutic agent. Ixabepilone "
            "achieves comparable pathologic complete response (pCR) rates "
            "to other active drugs in the neoadjuvant setting. "
            "This phase II trial was designed to investigate potential "
            "biomarkers that differentiate response to this agent."
        ),
        "journal": "American Association for Cancer Research",
        "year": 2013,
        "link": "https://pubmed.ncbi.nlm.nih.gov/23340299/"
    }
}


# ==================================================
# DATA LOADING
# ==================================================
def load_index():
    return pd.read_csv(DATA_INDEX_PATH)


def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_matrix(df):
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "feature"})
    if "feature" in df.columns:
        df = df.set_index("feature")
    return df.apply(pd.to_numeric, errors="coerce")


def load_dataset(index_df, gse, drug):
    row = index_df[(index_df.GSE == gse) & (index_df.Drug == drug)].iloc[0]

    clinical = read_table(row.clinical_path)
    expr = normalize_matrix(read_table(row.gene_path))
    pathway = normalize_matrix(read_table(row.pathway_path))

    clinical[SAMPLE_ID_COL] = clinical[SAMPLE_ID_COL].astype(str)
    expr.columns = expr.columns.astype(str)
    pathway.columns = pathway.columns.astype(str)

    common = sorted(set(clinical[SAMPLE_ID_COL]) & set(expr.columns))
    clinical = clinical[clinical[SAMPLE_ID_COL].isin(common)]
    expr = expr[common]
    pathway = pathway[common]

    clinical["response"] = (
        clinical[PCR_COLUMN]
        .astype(str)
        .str.split(":", n=1)
        .str[-1]
        .str.strip()
        .str.lower()
        .map({"yes": "Sensitive", "no": "Resistant"})
    )

    return clinical, expr, pathway


# ==================================================
# PCA / UMAP
# ==================================================
def make_pca(expr, clinical, selected):
    X = expr.T
    X = X[expr.var(axis=1).sort_values(ascending=False).head(2000).index]

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=X.index)
    df["sample"] = df.index
    df["response"] = clinical.set_index(SAMPLE_ID_COL).loc[df.index, "response"]

    sil = silhouette_score(
        X_scaled,
        df["response"].map({"Sensitive": 0, "Resistant": 1})
    )

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="response",
        hover_name="sample",
        title=f"PCA (Silhouette = {sil:.2f})"
    )

    if selected in df["sample"].values:
        sel = df[df["sample"] == selected]
        fig.add_scatter(
            x=sel.PC1, y=sel.PC2,
            marker=dict(size=16, symbol="star", color="black"),
            name="Selected sample"
        )

    return fig


def make_umap(expr, clinical, selected):
    X = expr.T
    X = X[expr.var(axis=1).sort_values(ascending=False).head(2000).index]

    X_scaled = StandardScaler().fit_transform(X)
    emb = umap.UMAP(random_state=42).fit_transform(X_scaled)

    df = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"], index=X.index)
    df["sample"] = df.index
    df["response"] = clinical.set_index(SAMPLE_ID_COL).loc[df.index, "response"]

    fig = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        color="response",
        hover_name="sample",
        title="UMAP"
    )

    if selected in df["sample"].values:
        sel = df[df["sample"] == selected]
        fig.add_scatter(
            x=sel.UMAP1, y=sel.UMAP2,
            marker=dict(size=16, symbol="star", color="black"),
            name="Selected sample"
        )

    return fig


# ==================================================
# HEATMAPS
# ==================================================
def static_clustermap_png(data, title):
    z = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]
    z = z.dropna()
    z = z.loc[data.var(axis=1).sort_values(ascending=False).head(60).index]

    cg = sns.clustermap(z, cmap="vlag", figsize=(7, 6))
    cg.fig.suptitle(title, y=1.02)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def interactive_heatmap(data, title, selected):
    z = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]
    z = z.dropna().loc[data.var(axis=1).sort_values(ascending=False).head(100).index]

    fig = px.imshow(
        z,
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r"
    )

    if selected in z.columns:
        fig.add_vline(x=list(z.columns).index(selected), line_width=3)

    return fig


# ==================================================
# DIFFERENTIAL EXPRESSION
# ==================================================
def compute_de(expr, clinical):
    log_expr = np.log2(expr + 1)
    meta = clinical.set_index(SAMPLE_ID_COL)

    sens = log_expr.loc[:, meta["response"] == "Sensitive"]
    res = log_expr.loc[:, meta["response"] == "Resistant"]

    log2fc = sens.mean(axis=1) - res.mean(axis=1)
    _, pvals = ttest_ind(sens.T, res.T, equal_var=False)

    df = pd.DataFrame({
        "gene": log_expr.index,
        "log2FC": log2fc,
        "neglog10_p": -np.log10(np.clip(pvals, 1e-300, 1)),
        "avg_expr": (sens.mean(axis=1) + res.mean(axis=1)) / 2
    })

    df["direction"] = np.where(
        df["log2FC"] >= 0, "Higher in Sensitive", "Higher in Resistant"
    )

    return df


def make_volcano(expr, clinical):
    df = compute_de(expr, clinical)
    return px.scatter(df, x="log2FC", y="neglog10_p", color="direction", hover_name="gene")


def make_ma(expr, clinical):
    df = compute_de(expr, clinical)
    return px.scatter(df, x="avg_expr", y="log2FC", color="direction", hover_name="gene")


# ==================================================
# STREAMLIT APP
# ==================================================
st.set_page_config(page_title="Clinical-Omics Explorer", layout="wide")
st.write("DEBUG page state:", st.session_state.get("page"))


if "page" not in st.session_state:
    st.session_state.page = "intro"


def render_intro():
    st.title("🧬 Clinical-Omics Explorer")

    gse = st.selectbox("Select Study (GSE)", list(STUDY_INFO.keys()))
    info = STUDY_INFO[gse]

    st.markdown(f"## {info['title']}")
    st.write(info["abstract"])

    st.markdown(
        f"**Journal:** {info['journal']}  \n"
        f"**Year:** {info['year']}  \n"
        f"[View Publication]({info['link']})"
    )

    st.divider()

    if st.button("➡️ Go to Analysis Dashboard", type="primary"):
        st.session_state.page = "dashboard"
        


def render_dashboard():

    # --- TOP LEFT BACK BUTTON (ALWAYS VISIBLE) ---
    with st.container():
        col_back, col_title = st.columns([1, 9])
        with col_back:
            if st.button("⬅️ Back"):
                st.session_state.page = "intro"
        with col_title:
            st.markdown("## 📊 Analysis Dashboard")
       


    index_df = load_index()

    @st.cache_data(show_spinner=True)
    def cached_dataset(gse, drug):
        return load_dataset(index_df, gse, drug)

    gse = st.selectbox("Select Study", sorted(index_df.GSE.unique()))
    drug = st.selectbox("Select Drug", index_df[index_df.GSE == gse].Drug.unique())

    clinical, expr, pathway = cached_dataset(gse, drug)

    sample = st.selectbox("Select Sample", ["ALL"] + list(expr.columns))
    selected = None if sample == "ALL" else sample

    plot = st.selectbox(
        "Select Plot",
        ["All plots", "PCA", "UMAP", "Gene Heatmap", "Pathway Heatmap", "Volcano", "MA"]
    )

    heatmap_mode = st.radio("Heatmap Mode", ["static", "interactive"], horizontal=True)

    def render_heatmap(data, title):
        if heatmap_mode == "static":
            st.image(static_clustermap_png(data, title), width="stretch")
        else:
            st.plotly_chart(
                interactive_heatmap(data, title, selected),
                width="stretch"
            )

    st.divider()

    if plot == "All plots":
        c1, c2 = st.columns(2)
        c1.plotly_chart(make_pca(expr, clinical, selected), width="stretch")
        c2.plotly_chart(make_umap(expr, clinical, selected), width="stretch")

        st.divider()
        render_heatmap(expr, "Gene Expression Heatmap")

        st.divider()
        render_heatmap(pathway, "Pathway Heatmap")

        st.divider()
        c3, c4 = st.columns(2)
        c3.plotly_chart(make_volcano(expr, clinical), width="stretch")
        c4.plotly_chart(make_ma(expr, clinical), width="stretch")

    elif plot == "PCA":
        st.plotly_chart(make_pca(expr, clinical, selected), width="stretch")
    elif plot == "UMAP":
        st.plotly_chart(make_umap(expr, clinical, selected), width="stretch")
    elif plot == "Gene Heatmap":
        render_heatmap(expr, "Gene Expression Heatmap")
    elif plot == "Pathway Heatmap":
        render_heatmap(pathway, "Pathway Heatmap")
    elif plot == "Volcano":
        st.plotly_chart(make_volcano(expr, clinical), width="stretch")
    elif plot == "MA":
        st.plotly_chart(make_ma(expr, clinical), width="stretch")


if st.session_state.page == "intro":
    render_intro()
else:
    render_dashboard()
