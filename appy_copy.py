import os
import io
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
import umap.umap_ as umap

# ---- matplotlib / seaborn for static heatmap ----
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
        "title": "Biomarker analysis of neoadjuvant Doxorubicin/Cyclophosphamide followed by Ixabepilone or Paclitaxel in early-stage breast cancer",
        "abstract": (
            "Predictive biomarkers offer the potential to improve the benefit:risk ratio of a therapeutic agent. "
            "Ixabepilone achieves comparable pathologic complete response (pCR) rates to other active drugs in the neoadjuvant setting. "
            "This phase II trial was designed to investigate potential biomarkers that differentiate response to this agent."
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
# PCA / UMAP (FIXED)
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

    counts = df["response"].value_counts().to_dict()
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
        color_discrete_map={
            "Sensitive": "blue",
            "Resistant": "red"
        },
        title=(
            f"PCA (n={len(df)})<br>"
            f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
            f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%<br>"
            f"Sensitive={counts.get('Sensitive',0)}, "
            f"Resistant={counts.get('Resistant',0)}, "
            f"Silhouette={sil:.2f}"
        )
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
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = umap_model.fit_transform(X_scaled)

    df = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"], index=X.index)
    df["sample"] = df.index
    df["response"] = clinical.set_index(SAMPLE_ID_COL).loc[df.index, "response"]

    sil = silhouette_score(
        X_scaled,
        df["response"].map({"Sensitive": 0, "Resistant": 1})
    )

    fig = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        color="response",
        hover_name="sample",
        color_discrete_map={
            "Sensitive": "blue",
            "Resistant": "red"
        },
        title=(
            f"UMAP (n={len(df)})<br>"
            f"n_neighbors=15, min_dist=0.1<br>"
            f"Silhouette={sil:.2f}"
        )
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
# HEATMAPS (STATIC + INTERACTIVE)
# ==================================================
def static_clustermap_png(data, title):
    z = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]
    z = z.dropna()

    top = data.var(axis=1).sort_values(ascending=False).head(60).index
    z = z.loc[top]

    cg = sns.clustermap(
        z,
        cmap="vlag",
        figsize=(7, 6)
    )
    cg.fig.suptitle(title, y=1.02, fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)

    return buf.getvalue()   # ✅ return PNG bytes



def interactive_heatmap(data, title, selected):
    z = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]
    z = z.dropna()

    top = data.var(axis=1).sort_values(ascending=False).head(100).index
    z = z.loc[top]

    fig = px.imshow(
        z,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=f"{title} (Interactive)"
    )

    if selected in z.columns:
        fig.add_vline(
            x=list(z.columns).index(selected),
            line_width=3,
            line_dash="dash",
            line_color="red"
        )

    fig.update_layout(height=600)
    return fig

# ==================================================
# DIFFERENTIAL EXPRESSION (MA & VOLCANO)
# ==================================================
def compute_de(expr, clinical):
    """
    Differential expression:
    log2FC = mean(Sensitive) - mean(Resistant)
    Direction is based ONLY on sign of log2FC (exploratory).
    """
    log_expr = np.log2(expr + 1)
    meta = clinical.set_index(SAMPLE_ID_COL)

    sens_mask = meta["response"] == "Sensitive"
    res_mask = meta["response"] == "Resistant"

    sens_expr = log_expr.loc[:, sens_mask]
    res_expr = log_expr.loc[:, res_mask]

    mean_sens = sens_expr.mean(axis=1)
    mean_res = res_expr.mean(axis=1)

    log2fc = mean_sens - mean_res

    # Welch t-test (still computed for reference)
    _, pvals = ttest_ind(
        sens_expr.T,
        res_expr.T,
        equal_var=False
    )

    df = pd.DataFrame({
        "gene": log_expr.index,
        "log2FC": log2fc,
        "pval": pvals,
        "neglog10_p": -np.log10(np.clip(pvals, 1e-300, 1)),
        "avg_expr": (mean_sens + mean_res) / 2
    })

    # Direction ONLY by sign
    df["direction"] = np.where(
        df["log2FC"] >= 0,
        "Higher in Sensitive",
        "Higher in Resistant"
    )

    return df



def make_volcano(expr, clinical):
    df = compute_de(expr, clinical)

    fig = px.scatter(
        df,
        x="log2FC",
        y="neglog10_p",
        color="direction",
        hover_name="gene",
        color_discrete_map={
            "Higher in Sensitive": "blue",
            "Higher in Resistant": "red"
        },
        title=(
            "Volcano Plot (Exploratory)<br>"
            "log2FC = mean(Sensitive) − mean(Resistant)"
        )
    )

    # Reference lines (optional, for context)
    fig.add_vline(x=0, line_dash="dash")
    fig.add_hline(y=-np.log10(0.05), line_dash="dot")

    fig.update_layout(height=500)
    return fig




def make_ma(expr, clinical):
    df = compute_de(expr, clinical)

    fig = px.scatter(
        df,
        x="avg_expr",
        y="log2FC",
        color="direction",
        hover_name="gene",
        color_discrete_map={
            "Higher in Sensitive": "blue",
            "Higher in Resistant": "red"
        },
        title="MA Plot (Exploratory)<br>log2FC = mean(Sensitive) - mean(Resistant)"
    )

    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=500)
    return fig





# ==================================================
# DASH APP
# ==================================================


index_df = load_index()


@st.cache_data(show_spinner=True)
def cached_dataset(gse, drug):
    return load_dataset(index_df, gse, drug)



# ==================================================
# ROUTER LAYOUT
# ==================================================
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content")
])


# ==================================================
# PAGE LAYOUTS
# ==================================================



# ==================================================
# CALLBACKS
# ==================================================


# ==================================================
# STREAMLIT MAIN APP
# ==================================================

st.set_page_config(
    page_title="Clinical-Omics Explorer",
    layout="wide"
)

st.title("🧬 Clinical-Omics Explorer")

# -------------------------------
# Load index
# -------------------------------
index_df = load_index()

# -------------------------------
# Study selector
# -------------------------------
gse = st.selectbox(
    "Select Study (GSE)",
    sorted(index_df.GSE.unique())
)

study_info = STUDY_INFO[gse]

st.markdown(f"### {study_info['title']}")
st.write(study_info["abstract"])

st.markdown(
    f"**Journal:** {study_info['journal']}  \n"
    f"**Year:** {study_info['year']}  \n"
    f"[View Publication]({study_info['link']})"
)

st.divider()

# -------------------------------
# Drug selector
# -------------------------------
drug = st.selectbox(
    "Select Drug",
    index_df[index_df.GSE == gse].Drug.unique()
)

# -------------------------------
# Load data (cached)
# -------------------------------
clinical, expr, pathway = cached_dataset(gse, drug)

# -------------------------------
# Sample selector
# -------------------------------
samples = ["ALL"] + list(expr.columns)
sample = st.selectbox("Select Sample", samples)
selected = None if sample == "ALL" else sample

# -------------------------------
# Plot selector
# -------------------------------
plot = st.selectbox(
    "Select Plot",
    [
        "All plots",
        "PCA",
        "UMAP",
        "Gene Heatmap",
        "Pathway Heatmap",
        "Volcano",
        "MA"
    ]
)

# -------------------------------
# Heatmap mode
# -------------------------------
heatmap_mode = st.radio(
    "Heatmap Mode",
    ["static", "interactive"],
    horizontal=True
)

st.divider()

def render_heatmap(data, title):
    if heatmap_mode == "static":
        png = static_clustermap_png(data, title)
        st.image(png, caption=title, use_container_width=True)
    else:
        fig = interactive_heatmap(data, title, selected)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Render chosen plot(s)
# -------------------------------
if plot == "All plots":
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_pca(expr, clinical, selected), use_container_width=True)
    with c2:
        st.plotly_chart(make_umap(expr, clinical, selected), use_container_width=True)

    st.divider()
    st.subheader("Gene Expression")
    render_heatmap(expr, "Gene Expression Heatmap")

    st.divider()
    st.subheader("Pathway Activity")
    render_heatmap(pathway, "Pathway Heatmap")

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(make_volcano(expr, clinical), use_container_width=True)
    with c4:
        st.plotly_chart(make_ma(expr, clinical), use_container_width=True)

elif plot == "PCA":
    st.plotly_chart(make_pca(expr, clinical, selected), use_container_width=True)

elif plot == "UMAP":
    st.plotly_chart(make_umap(expr, clinical, selected), use_container_width=True)

elif plot == "Gene Heatmap":
    render_heatmap(expr, "Gene Expression Heatmap")

elif plot == "Pathway Heatmap":
    render_heatmap(pathway, "Pathway Heatmap")

elif plot == "Volcano":
    st.plotly_chart(make_volcano(expr, clinical), use_container_width=True)

elif plot == "MA":
    st.plotly_chart(make_ma(expr, clinical), use_container_width=True)




