# ==================================================
# CLINICAL-OMICS EXPLORER - REFINED EDITION v4.1
# ==================================================
# Refined UX with genomics-focused design and enhanced analytics
# ==================================================

from ast import expr
import io
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
import umap.umap_ as umap

import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# PDF generation (optional)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors as pdf_colors
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ==================================================
# CONFIGURATION
# ==================================================
APP_PASSWORD = "clinical2024"
SESSIONS_DIR = "saved_sessions"
THEME_FILE = "user_theme.json"

RESPONSE_COLORS = {"Sensitive": "#1f77b4", "Resistant": "#d62728"}
DIRECTION_COLORS = {"Higher in Sensitive": "#1f77b4", "Higher in Resistant": "#d62728"}

DATA_INDEX_PATH = "data/index.csv"
SAMPLE_ID_COL = "!Sample_geo_accession"
PCR_COLUMN = "!Sample_characteristics_ch1.4"

STUDY_INFO = {
    "GSE41998": {
        "title": "Biomarker analysis of neoadjuvant Doxorubicin/Cyclophosphamide followed by Ixabepilone or Paclitaxel in early-stage breast cancer",
        "abstract": "Predictive biomarkers offer the potential to improve the benefit:risk ratio of a therapeutic agent. Ixabepilone achieves comparable pathologic complete response (pCR) rates to other active drugs in the neoadjuvant setting. This phase II trial was designed to investigate potential biomarkers that differentiate response to this agent.",
        "journal": "American Association for Cancer Research",
        "year": 2013,
        "link": "https://pubmed.ncbi.nlm.nih.gov/23340299/"
    }
}

# ==================================================
# THEME MANAGEMENT
# ==================================================
def load_theme():
    if os.path.exists(THEME_FILE):
        try:
            with open(THEME_FILE, 'r') as f:
                return json.load(f).get('theme', 'light')
        except:
            return 'light'
    return 'light'

def save_theme(theme):
    with open(THEME_FILE, 'w') as f:
        json.dump({'theme': theme}, f)

# ==================================================
# SESSION MANAGEMENT
# ==================================================
def get_session_list():
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)
        return []
    return sorted([f.replace('.json', '') for f in os.listdir(SESSIONS_DIR) if f.endswith('.json')], reverse=True)

def save_session(name, data):
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)
    session_data = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'study_id': data.get('study_id'),
        'drug': data.get('drug'),
        'sample': data.get('sample'),
        'pval_threshold': data.get('pval_threshold', 0.05),
        'fc_threshold': data.get('fc_threshold', 0.5),
        'notes': data.get('notes', '')
    }
    with open(f"{SESSIONS_DIR}/{name}.json", 'w') as f:
        json.dump(session_data, f, indent=2)
    return True

def load_session(name):
    filename = f"{SESSIONS_DIR}/{name}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def delete_session(name):
    filename = f"{SESSIONS_DIR}/{name}.json"
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False


# ==================================================
# CUSTOM CSS WITH THEME SUPPORT
# ==================================================
def apply_custom_css(theme='light'):
    if theme == 'dark':
        bg, text, card_bg, border = "#1e1e1e", "#ffffff", "#2d2d2d", "#404040"
        sidebar_bg, header_grad = "#252525", "linear-gradient(135deg, #4a5568 0%, #2d3748 100%)"
    else:
        bg, text, card_bg, border = "white", "#000000", "#f8f9fa", "#dee2e6"
        sidebar_bg, header_grad = "#f8f9fa", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    
    st.markdown(f"""
        <style>
        .main {{ background-color: {bg}; color: {text}; }}
        [data-testid="stSidebar"] {{ background-color: {sidebar_bg}; }}
        .stApp {{ background-color: {bg}; }}
        .sticky-header {{
            position: sticky; top: 0; z-index: 999;
            background: {header_grad}; padding: 1rem; color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem;
        }}
        .progress-container {{
            width: 100%; background-color: rgba(255,255,255,0.2);
            border-radius: 10px; height: 8px; margin-top: 10px;
        }}
        .progress-bar {{
            height: 100%; border-radius: 10px; background-color: #4CAF50;
            transition: width 0.3s ease;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem; border-radius: 10px; color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 0.5rem 0;
        }}
        .metric-card-blue {{ background: linear-gradient(135deg, #1f77b4 0%, #3498db 100%); }}
        .metric-card-red {{ background: linear-gradient(135deg, #d62728 0%, #e74c3c 100%); }}
        .metric-card-green {{ background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); }}
        .metric-card-orange {{ background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%); }}
        .metric-value {{ font-size: 2.5rem; font-weight: bold; margin: 0; }}
        .metric-label {{ font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem; }}
        .filter-panel {{ background-color: {card_bg}; padding: 1.5rem; border-radius: 10px; border: 1px solid {border}; margin-bottom: 1rem; }}
        .breadcrumb {{ display: flex; align-items: center; gap: 0.5rem; color: white; font-size: 0.9rem; }}
        .breadcrumb-item {{ opacity: 0.7; }}
        .breadcrumb-item.active {{ opacity: 1; font-weight: bold; }}
        .computation-panel {{
            background-color: {card_bg};
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }}
        </style>
    """, unsafe_allow_html=True)

# ==================================================
# UTILITY FUNCTIONS
# ==================================================
def get_progress_percentage(page):
    pages = ["login", "study_selection", "intro", "dashboard"]
    try:
        return (pages.index(page) + 1) / len(pages) * 100
    except:
        return 0

def fig_to_bytes(fig, format='png'):
    try:
        return fig.to_image(format=format, width=1200, height=800)
    except:
        return None

# ==================================================
# DATA LOADING
# ==================================================
def load_index():
    if os.path.exists(DATA_INDEX_PATH):
        return pd.read_csv(DATA_INDEX_PATH)
    return pd.DataFrame(columns=['GSE', 'Drug', 'clinical_path', 'gene_path', 'pathway_path'])

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
        clinical[PCR_COLUMN].astype(str).str.split(":", n=1).str[-1]
        .str.strip().str.lower().map({"yes": "Sensitive", "no": "Resistant"})
    )
    return clinical, expr, pathway

# ==================================================
# ANALYSIS FUNCTIONS
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
    df["direction"] = np.where(df["log2FC"] >= 0, "Higher in Sensitive", "Higher in Resistant")
    df["p_value"] = 10 ** (-df["neglog10_p"])
    return df

def compute_summary_stats(clinical, expr):
    total = len(clinical)
    sens = (clinical['response'] == 'Sensitive').sum()
    res = (clinical['response'] == 'Resistant').sum()
    
    de_df = compute_de(expr, clinical)
    sig = (de_df['neglog10_p'] > -np.log10(0.05)).sum()
    top = de_df.nlargest(1, 'neglog10_p').iloc[0]
    
    return {
        'total_samples': total,
        'sensitive_count': sens,
        'resistant_count': res,
        'sensitive_pct': (sens / total * 100) if total > 0 else 0,
        'resistant_pct': (res / total * 100) if total > 0 else 0,
        'significant_genes': sig,
        'top_gene': top['gene'],
        'top_gene_pval': 10 ** (-top['neglog10_p'])
    }

# ==================================================
# NEW: STUDY-DERIVED COMPUTATIONS
# ==================================================
def compute_study_metrics(expr, clinical, pathway):
    """
    Compute derived metrics from the study data
    """
    metrics = {}
    
    # Gene expression metrics
    log_expr = np.log2(expr + 1)
    metrics['mean_gene_expression'] = log_expr.mean().mean()
    metrics['median_gene_expression'] = log_expr.median().median()
    metrics['gene_expression_variance'] = log_expr.var().mean()
    
    # Sample-wise metrics
    meta = clinical.set_index(SAMPLE_ID_COL)
    sens_samples = meta[meta["response"] == "Sensitive"].index
    res_samples = meta[meta["response"] == "Resistant"].index
    
    metrics['sensitive_mean_expr'] = log_expr[sens_samples].mean().mean()
    metrics['resistant_mean_expr'] = log_expr[res_samples].mean().mean()
    metrics['expr_difference'] = metrics['sensitive_mean_expr'] - metrics['resistant_mean_expr']
    
    # Pathway metrics
    log_pathway = np.log2(pathway + 1)
    metrics['mean_pathway_activity'] = log_pathway.mean().mean()
    metrics['pathway_variance'] = log_pathway.var().mean()
    
    # Differential expression metrics
    de_df = compute_de(expr, clinical)
    metrics['total_genes'] = len(de_df)
    metrics['sig_genes_p005'] = (de_df['p_value'] < 0.05).sum()
    metrics['sig_genes_p001'] = (de_df['p_value'] < 0.01).sum()
    metrics['high_fc_genes'] = (abs(de_df['log2FC']) > 1).sum()
    metrics['very_high_fc_genes'] = (abs(de_df['log2FC']) > 2).sum()
    
    # Expression distribution
    metrics['genes_higher_sensitive'] = (de_df['direction'] == 'Higher in Sensitive').sum()
    metrics['genes_higher_resistant'] = (de_df['direction'] == 'Higher in Resistant').sum()
    
    return metrics

def compute_sample_profile(sample_id, expr, clinical, pathway):
    """
    Compute comprehensive profile for a specific sample
    """
    if sample_id not in expr.columns:
        return None
    
    profile = {}
    
    # Basic info
    meta = clinical.set_index(SAMPLE_ID_COL)
    profile['sample_id'] = sample_id
    profile['response'] = meta.loc[sample_id, 'response']
    
    # Gene expression stats
    sample_expr = np.log2(expr[sample_id] + 1)
    profile['mean_expression'] = sample_expr.mean()
    profile['median_expression'] = sample_expr.median()
    profile['max_expression'] = sample_expr.max()
    profile['min_expression'] = sample_expr.min()
    
    # Top expressed genes
    profile['top_5_genes'] = sample_expr.nlargest(5).to_dict()
    
    # Pathway activity
    sample_pathway = np.log2(pathway[sample_id] + 1)
    profile['mean_pathway_activity'] = sample_pathway.mean()
    profile['top_5_pathways'] = sample_pathway.nlargest(5).to_dict()
    
    # Comparison to group
    group_samples = meta[meta['response'] == profile['response']].index
    group_expr = np.log2(expr[group_samples] + 1).mean(axis=1)
    
    profile['vs_group_correlation'] = sample_expr.corr(group_expr)
    profile['deviation_from_group'] = (sample_expr - group_expr).abs().mean()
    
    return profile

def compute_gene_profile(gene_name, expr, clinical):
    """
    Compute comprehensive profile for a specific gene
    """
    if gene_name not in expr.index:
        return None
    
    profile = {}
    profile['gene_name'] = gene_name
    
    # Expression stats
    gene_expr = np.log2(expr.loc[gene_name] + 1)
    profile['mean_expression'] = gene_expr.mean()
    profile['median_expression'] = gene_expr.median()
    profile['std_expression'] = gene_expr.std()
    profile['cv'] = (gene_expr.std() / gene_expr.mean()) * 100 if gene_expr.mean() > 0 else 0
    
    # By response group
    meta = clinical.set_index(SAMPLE_ID_COL)
    sens_samples = meta[meta["response"] == "Sensitive"].index
    res_samples = meta[meta["response"] == "Resistant"].index
    
    profile['sensitive_mean'] = gene_expr[sens_samples].mean()
    profile['resistant_mean'] = gene_expr[res_samples].mean()
    profile['fold_change'] = profile['sensitive_mean'] - profile['resistant_mean']
    
    # Statistical test
    _, pval = ttest_ind(gene_expr[sens_samples], gene_expr[res_samples], equal_var=False)
    profile['p_value'] = pval
    profile['significant'] = pval < 0.05
    
    return profile

def compute_pathway_profile(pathway_name, pathway, clinical):
    """
    Compute comprehensive profile for a specific pathway
    """
    if pathway_name not in pathway.index:
        return None
    
    profile = {}
    profile['pathway_name'] = pathway_name
    
    # Activity stats
    pathway_activity = np.log2(pathway.loc[pathway_name] + 1)
    profile['mean_activity'] = pathway_activity.mean()
    profile['median_activity'] = pathway_activity.median()
    profile['std_activity'] = pathway_activity.std()
    
    # By response group
    meta = clinical.set_index(SAMPLE_ID_COL)
    sens_samples = meta[meta["response"] == "Sensitive"].index
    res_samples = meta[meta["response"] == "Resistant"].index
    
    profile['sensitive_mean'] = pathway_activity[sens_samples].mean()
    profile['resistant_mean'] = pathway_activity[res_samples].mean()
    profile['activity_difference'] = profile['sensitive_mean'] - profile['resistant_mean']
    
    # Statistical test
    _, pval = ttest_ind(pathway_activity[sens_samples], pathway_activity[res_samples], equal_var=False)
    profile['p_value'] = pval
    profile['significant'] = pval < 0.05
    
    return profile

# ==================================================
# AUTO-INSIGHTS
# ==================================================
def generate_auto_insights(stats, de_df, clinical, expr):
    insights = []
    
    # Dataset balance
    if stats['sensitive_pct'] > 70:
        insights.append({'type': 'warning', 'title': 'Imbalanced Dataset', 
                        'text': f"Dataset skewed towards Sensitive ({stats['sensitive_pct']:.1f}%). Consider this in interpretation."})
    elif stats['resistant_pct'] > 70:
        insights.append({'type': 'warning', 'title': 'Imbalanced Dataset',
                        'text': f"Dataset skewed towards Resistant ({stats['resistant_pct']:.1f}%). Consider this in interpretation."})
    else:
        insights.append({'type': 'success', 'title': 'Balanced Dataset',
                        'text': f"Good distribution: {stats['sensitive_pct']:.1f}% Sensitive vs {stats['resistant_pct']:.1f}% Resistant."})
    
    # Differential expression
    sig_pct = (stats['significant_genes'] / len(de_df) * 100) if len(de_df) > 0 else 0
    if sig_pct > 20:
        insights.append({'type': 'success', 'title': 'Strong Differential Expression',
                        'text': f"Found {stats['significant_genes']} significant genes ({sig_pct:.1f}% of total). Strong signal!"})
    elif sig_pct > 5:
        insights.append({'type': 'info', 'title': 'Moderate Differential Expression',
                        'text': f"Found {stats['significant_genes']} significant genes ({sig_pct:.1f}% of total)."})
    else:
        insights.append({'type': 'warning', 'title': 'Weak Differential Expression',
                        'text': f"Only {stats['significant_genes']} significant genes ({sig_pct:.1f}% of total)."})
    
    # Top biomarkers
    top_genes = de_df.nsmallest(5, 'p_value')['gene'].tolist()
    insights.append({'type': 'info', 'title': 'Top Biomarker Candidates',
                    'text': f"Most significant genes: {', '.join(top_genes[:3])}. Consider for validation."})
    
    # High fold changes
    high_fc = de_df[abs(de_df['log2FC']) > 2]
    if len(high_fc) > 10:
        insights.append({'type': 'success', 'title': 'High Magnitude Changes',
                        'text': f"Found {len(high_fc)} genes with |log2FC| > 2. Strong biological effects."})
    
    # Directionality
    up_sens = (de_df['direction'] == 'Higher in Sensitive').sum()
    up_res = (de_df['direction'] == 'Higher in Resistant').sum()
    if up_sens > up_res * 1.5:
        insights.append({'type': 'info', 'title': 'Expression Pattern',
                        'text': f"More genes upregulated in Sensitive ({up_sens} vs {up_res}). May indicate response pathways."})
    elif up_res > up_sens * 1.5:
        insights.append({'type': 'info', 'title': 'Expression Pattern',
                        'text': f"More genes upregulated in Resistant ({up_res} vs {up_sens}). May indicate resistance mechanisms."})
    
    return insights


# ==================================================
# PDF GENERATION
# ==================================================
def generate_pdf_report(study_id, drug, stats, de_df):
    if not PDF_AVAILABLE:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                     fontSize=24, textColor=pdf_colors.HexColor('#667eea'), spaceAfter=30)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                       fontSize=16, textColor=pdf_colors.HexColor('#764ba2'), spaceAfter=12)
        
        # Title page
        story.append(Paragraph(f"Clinical-Omics Analysis Report", title_style))
        story.append(Paragraph(f"Study: {study_id}", styles['Heading2']))
        story.append(Paragraph(f"Drug: {drug}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Study info
        if study_id in STUDY_INFO:
            info = STUDY_INFO[study_id]
            story.append(Paragraph("Study Information", heading_style))
            story.append(Paragraph(f"<b>Title:</b> {info['title']}", styles['Normal']))
            story.append(Paragraph(f"<b>Journal:</b> {info['journal']}", styles['Normal']))
            story.append(Paragraph(f"<b>Year:</b> {info['year']}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
        
        # Summary stats
        story.append(Paragraph("Summary Statistics", heading_style))
        stats_data = [
            ['Metric', 'Value'],
            ['Total Samples', str(stats['total_samples'])],
            ['Sensitive Samples', f"{stats['sensitive_count']} ({stats['sensitive_pct']:.1f}%)"],
            ['Resistant Samples', f"{stats['resistant_count']} ({stats['resistant_pct']:.1f}%)"],
            ['Significant Genes (p<0.05)', str(stats['significant_genes'])],
            ['Top Differential Gene', stats['top_gene']],
            ['Top Gene P-value', f"{stats['top_gene_pval']:.2e}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pdf_colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), pdf_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), pdf_colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, pdf_colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Top genes
        story.append(PageBreak())
        story.append(Paragraph("Top 20 Differentially Expressed Genes", heading_style))
        top_de = de_df.nsmallest(20, 'p_value')[['gene', 'log2FC', 'p_value', 'direction']]
        top_de['p_value'] = top_de['p_value'].apply(lambda x: f"{x:.2e}")
        top_de['log2FC'] = top_de['log2FC'].apply(lambda x: f"{x:.3f}")
        
        de_data = [['Gene', 'Log2FC', 'P-value', 'Direction']]
        for _, row in top_de.iterrows():
            de_data.append([row['gene'], row['log2FC'], row['p_value'], row['direction']])
        
        de_table = Table(de_data, colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 2*inch])
        de_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pdf_colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), pdf_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, pdf_colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [pdf_colors.white, pdf_colors.lightgrey])
        ]))
        story.append(de_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# ==================================================
# ENHANCED CHATBOT
# ==================================================
def process_enhanced_chatbot(command, expr, clinical, pathway, selected, study_id):
    cmd = command.lower().strip()
    
    # Gene expression query
    if "expression of" in cmd or "expression for" in cmd:
        words = cmd.split()
        for i, word in enumerate(words):
            if word in ["of", "for"] and i + 1 < len(words):
                gene_name = words[i + 1].upper().strip('?.,')
                if gene_name in expr.index:
                    gene_expr = expr.loc[gene_name]
                    mean_expr = gene_expr.mean()
                    sens_expr = gene_expr[clinical.set_index(SAMPLE_ID_COL)['response'] == 'Sensitive'].mean()
                    res_expr = gene_expr[clinical.set_index(SAMPLE_ID_COL)['response'] == 'Resistant'].mean()
                    response = f"""**{gene_name} Expression Analysis:**
- Mean expression: {mean_expr:.3f}
- Sensitive samples: {sens_expr:.3f}
- Resistant samples: {res_expr:.3f}
- Fold change: {sens_expr / (res_expr + 0.001):.3f}x"""
                    return "text", None, response
                else:
                    return "text", None, f"Gene '{gene_name}' not found in dataset."
    
    # Sample count
    elif "how many" in cmd and "sample" in cmd:
        total = len(clinical)
        sens = (clinical['response'] == 'Sensitive').sum()
        res = (clinical['response'] == 'Resistant').sum()
        response = f"""**Sample Summary:**
- Total samples: {total}
- Sensitive: {sens} ({sens/total*100:.1f}%)
- Resistant: {res} ({res/total*100:.1f}%)"""
        return "text", None, response
    
    # Top genes
    elif "top" in cmd and ("gene" in cmd or "biomarker" in cmd):
        n = 10
        words = cmd.split()
        for i, word in enumerate(words):
            if word == "top" and i + 1 < len(words):
                try:
                    n = int(words[i + 1])
                except:
                    pass
        de_df = compute_de(expr, clinical)
        top_genes = de_df.nsmallest(n, 'p_value')
        response = f"**Top {n} Differentially Expressed Genes:**\n\n"
        for _, row in top_genes.iterrows():
            response += f"- **{row['gene']}**: FC={row['log2FC']:.3f}, p={row['p_value']:.2e}\n"
        return "text", None, response
    
    # Summary
    elif "summary" in cmd or "overview" in cmd:
        stats = compute_summary_stats(clinical, expr)
        response = f"""**Analysis Summary for {study_id}:**
- Total samples: {stats['total_samples']}
- Sensitive: {stats['sensitive_count']} ({stats['sensitive_pct']:.1f}%)
- Resistant: {stats['resistant_count']} ({stats['resistant_pct']:.1f}%)
- Significant genes: {stats['significant_genes']}
- Top gene: {stats['top_gene']} (p={stats['top_gene_pval']:.2e})"""
        return "text", None, response
    
    # Comparison
    elif "compare" in cmd or "difference" in cmd:
        de_df = compute_de(expr, clinical)
        sig_genes = de_df[de_df['p_value'] < 0.05]
        up_sens = (sig_genes['direction'] == 'Higher in Sensitive').sum()
        up_res = (sig_genes['direction'] == 'Higher in Resistant').sum()
        response = f"""**Sensitive vs Resistant Comparison:**
- Genes higher in Sensitive: {up_sens}
- Genes higher in Resistant: {up_res}
- Ratio: {up_sens/max(up_res, 1):.2f}:1

This suggests {'Sensitive samples show more gene activation' if up_sens > up_res else 'Resistant samples show more gene activation'}."""
        return "text", None, response
    
    # Help
    elif "help" in cmd:
        return "text", None, """**Enhanced Chatbot Commands:**

**Data Queries:**
- `how many samples` - Get sample counts
- `summary` or `overview` - Get analysis summary
- `top 10 genes` - Show top differential genes
- `expression of [GENE]` - Get gene expression
- `compare sensitive resistant` - Compare groups

**Visualizations:**
- `show pca` - Display PCA plot
- `volcano plot` - Display volcano plot

Try natural language: "What's the expression of BRCA1?" or "Show me top 5 biomarkers"
"""
    
    # Fallback to plots
    elif "pca" in cmd:
        return "plot", make_pca(expr, clinical, selected), "PCA plot generated!"
    elif "umap" in cmd:
        return "plot", make_umap(expr, clinical, selected), "UMAP plot generated!"
    elif "volcano" in cmd:
        return "plot", make_volcano(expr, clinical), "Volcano plot generated!"
    
    return "text", None, "I didn't understand that. Type 'help' for available commands!"

def process_chatbot_command_study_search(command, index_df):
    cmd = command.lower().strip()
    if "search" in cmd or "find" in cmd or "show" in cmd or "list" in cmd:
        if "study" in cmd or "studies" in cmd or "gse" in cmd:
            if not index_df.empty:
                studies_list = index_df['GSE'].unique()
                response = "**Available Studies:**\n\n"
                for study in studies_list:
                    drugs = index_df[index_df['GSE'] == study]['Drug'].unique()
                    response += f"- **{study}**: {', '.join(drugs)}\n"
                return "text", None, response
            else:
                return "text", None, "No studies found."
    elif "gse" in cmd:
        words = cmd.split()
        for word in words:
            if word.startswith("gse"):
                study_id = word.upper()
                if study_id in index_df['GSE'].values:
                    drugs = index_df[index_df['GSE'] == study_id]['Drug'].unique()
                    return "text", None, f"✅ **{study_id}** found! Available drugs: {', '.join(drugs)}"
                else:
                    return "text", None, f"❌ **{study_id}** not found."
    elif "help" in cmd:
        return "text", None, """**Available Commands:**
- `search studies` or `list studies` - Show all studies
- `GSE41998` - Check if study exists
- `help` - Show this message"""
    return "text", None, "Type 'help' for commands or enter a study ID (e.g., GSE41998)."


# ==================================================
# VISUALIZATION FUNCTIONS
# ==================================================
def make_pca(expr, clinical, selected, highlight_gene=None):
    X = expr.T
    X = X[expr.var(axis=1).sort_values(ascending=False).head(2000).index]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    
    df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=X.index)
    df["sample"] = df.index
    df["response"] = clinical.set_index(SAMPLE_ID_COL).loc[df.index, "response"]
    
    # Add gene expression if specified
    if highlight_gene and highlight_gene in expr.index:
        gene_values = np.log2(expr.loc[highlight_gene, df.index] + 1)
        df[f"{highlight_gene}_expr"] = gene_values
        hover_data = {"sample": True, "response": True, f"{highlight_gene}_expr": ":.3f"}
        color = f"{highlight_gene}_expr"
        color_scale = "Viridis"
        title_suffix = f" (colored by {highlight_gene})"
    else:
        hover_data = {"sample": True, "response": True}
        color = "response"
        color_scale = None
        title_suffix = ""
    
    sil = silhouette_score(X_scaled, df["response"].map({"Sensitive": 0, "Resistant": 1}))
    
    if color_scale:
        fig = px.scatter(df, x="PC1", y="PC2", color=color, hover_name="sample",
                        hover_data=hover_data, title=f"PCA (Silhouette = {sil:.3f}){title_suffix}",
                        color_continuous_scale=color_scale)
    else:
        fig = px.scatter(df, x="PC1", y="PC2", color=color, hover_name="sample",
                        hover_data=hover_data, title=f"PCA (Silhouette = {sil:.3f}){title_suffix}",
                        color_discrete_map=RESPONSE_COLORS)
    
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12))
    
    if selected and selected in df["sample"].values:
        sel = df[df["sample"] == selected]
        fig.add_scatter(x=sel.PC1, y=sel.PC2, marker=dict(size=20, symbol="star", 
                       color="gold", line=dict(color="black", width=2)), 
                       name=f"Selected: {selected}", showlegend=True)
    return fig

def make_umap(expr, clinical, selected, highlight_gene=None):
    X = expr.T
    X = X[expr.var(axis=1).sort_values(ascending=False).head(2000).index]
    X_scaled = StandardScaler().fit_transform(X)
    emb = umap.UMAP(random_state=42).fit_transform(X_scaled)
    
    df = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"], index=X.index)
    df["sample"] = df.index
    df["response"] = clinical.set_index(SAMPLE_ID_COL).loc[df.index, "response"]
    
    # Add gene expression if specified
    if highlight_gene and highlight_gene in expr.index:
        gene_values = np.log2(expr.loc[highlight_gene, df.index] + 1)
        df[f"{highlight_gene}_expr"] = gene_values
        hover_data = {"sample": True, "response": True, f"{highlight_gene}_expr": ":.3f"}
        color = f"{highlight_gene}_expr"
        color_scale = "Viridis"
        title_suffix = f" (colored by {highlight_gene})"
    else:
        hover_data = {"sample": True, "response": True}
        color = "response"
        color_scale = None
        title_suffix = ""
    
    if color_scale:
        fig = px.scatter(df, x="UMAP1", y="UMAP2", color=color, hover_name="sample",
                        hover_data=hover_data, title=f"UMAP{title_suffix}",
                        color_continuous_scale=color_scale)
    else:
        fig = px.scatter(df, x="UMAP1", y="UMAP2", color=color, hover_name="sample",
                        hover_data=hover_data, title=f"UMAP{title_suffix}",
                        color_discrete_map=RESPONSE_COLORS)
    
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12))
    
    if selected and selected in df["sample"].values:
        sel = df[df["sample"] == selected]
        fig.add_scatter(x=sel.UMAP1, y=sel.UMAP2, marker=dict(size=20, symbol="star",
                       color="gold", line=dict(color="black", width=2)),
                       name=f"Selected: {selected}", showlegend=True)
    return fig

def make_volcano(expr, clinical):
    df = compute_de(expr, clinical)
    fig = px.scatter(df, x="log2FC", y="neglog10_p", color="direction", hover_name="gene",
                     color_discrete_map=DIRECTION_COLORS, hover_data={'log2FC': ':.3f', 'neglog10_p': ':.3f'})
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12))
    return fig

def make_ma(expr, clinical):
    df = compute_de(expr, clinical)
    fig = px.scatter(df, x="avg_expr", y="log2FC", color="direction", hover_name="gene",
                     color_discrete_map=DIRECTION_COLORS, hover_data={'avg_expr': ':.3f', 'log2FC': ':.3f'})
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12))
    return fig

def build_binary_clinical_matrix(clinical, expr_columns):
    clin = clinical.set_index(SAMPLE_ID_COL).loc[expr_columns]
    binary = {
        "response_Resistant": clin["response"].astype(str).str.lower().eq("resistant").astype(int),
        "response_Sensitive": clin["response"].astype(str).str.lower().eq("sensitive").astype(int),
    }
    return pd.DataFrame(binary, index=expr_columns).T

def static_binary_clinical_heatmap(clinical, expr):
    mat = build_binary_clinical_matrix(clinical, expr.columns)
    cmap = ListedColormap(["#d62728", "#1f77b4"])
    plt.figure(figsize=(14, 2.5))
    plt.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=90)
    plt.colorbar(ticks=[0, 1], label="Response")
    plt.xlabel("Samples")
    plt.ylabel("Response")
    plt.title("Clinical Response Heatmap")
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def interactive_binary_clinical_heatmap(clinical, expr, selected):
    mat = build_binary_clinical_matrix(clinical, expr.columns)
    fig = px.imshow(mat, aspect="auto", color_continuous_scale=[[0.0, "#d62728"], [1.0, "#1f77b4"]],
                    title="Clinical Response Heatmap")
    if selected and selected in mat.columns:
        fig.add_vline(x=list(mat.columns).index(selected), line_width=3, line_color="gold")
    fig.update_layout(xaxis_title="Samples", yaxis_title="Response", plot_bgcolor='white', paper_bgcolor='white')
    return fig

def static_clustermap_png(data, title):
    z = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]
    z = z.dropna().loc[data.var(axis=1).sort_values(ascending=False).head(60).index]
    cg = sns.clustermap(z, cmap="vlag", figsize=(7, 6))
    cg.fig.suptitle(title, y=1.02)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def interactive_heatmap(data, title, selected, highlight_feature=None):
    """
    Enhanced heatmap with optional feature highlighting
    """
    z = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]
    z = z.dropna()
    
    # If specific feature is requested, show it prominently
    if highlight_feature and highlight_feature in z.index:
        # Get top variable features
        top_features = data.var(axis=1).sort_values(ascending=False).head(99).index
        # Ensure highlighted feature is included
        if highlight_feature not in top_features:
            top_features = list(top_features[:98]) + [highlight_feature]
        else:
            top_features = list(top_features)
        z = z.loc[top_features]
    else:
        z = z.loc[data.var(axis=1).sort_values(ascending=False).head(100).index]
    
    fig = px.imshow(z, aspect="auto", title=title, color_continuous_scale="RdBu_r")
    
    # Highlight selected sample
    if selected and selected in z.columns:
        fig.add_vline(x=list(z.columns).index(selected), line_width=3, line_color="gold")
    
    # Highlight feature if specified
    if highlight_feature and highlight_feature in z.index:
        fig.add_hline(y=list(z.index).index(highlight_feature), line_width=3, line_color="lime")
    
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    return fig

# ==================================================
# PAGE COMPONENTS
# ==================================================
def render_sticky_header(page_name, study_id=None):
    progress = get_progress_percentage(page_name)
    breadcrumb_map = {
        "login": ["Login"],
        "study_selection": ["Login", "Study Selection"],
        "intro": ["Login", "Study Selection", "Study Info"],
        "dashboard": ["Login", "Study Selection", "Study Info", "Analysis"]
    }
    breadcrumbs = breadcrumb_map.get(page_name, [])
    breadcrumb_html = " → ".join([
        f'<span class="breadcrumb-item {"active" if i == len(breadcrumbs)-1 else ""}">{item}</span>'
        for i, item in enumerate(breadcrumbs)
    ])
    title = study_id if study_id else "Clinical-Omics Explorer"
    st.markdown(f"""
        <div class="sticky-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: white;">🧬 {title}</h2>
                    <div class="breadcrumb">{breadcrumb_html}</div>
                </div>
                <div style="text-align: right; font-size: 0.9rem;">Progress: {progress:.0f}%</div>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_metric_cards(stats):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card metric-card-blue">
            <div class="metric-value">{stats['total_samples']}</div>
            <div class="metric-label">Total Samples</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card metric-card-green">
            <div class="metric-value">{stats['sensitive_count']}</div>
            <div class="metric-label">Sensitive ({stats['sensitive_pct']:.1f}%)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card metric-card-red">
            <div class="metric-value">{stats['resistant_count']}</div>
            <div class="metric-label">Resistant ({stats['resistant_pct']:.1f}%)</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card metric-card-orange">
            <div class="metric-value">{stats['significant_genes']}</div>
            <div class="metric-label">Significant Genes (p<0.05)</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ==================================================
# PAGE RENDERING
# ==================================================
def render_login():
    render_sticky_header("login")
    # CHANGE 1: Gene/DNA symbol instead of lock
    st.title("🧬 Clinical-Omics Explorer")
    st.markdown("### Computational Biology & Genomics Analysis Platform")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Password", type="password", key="password_input")
        if st.button("Login", type="primary", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.page = "study_selection"
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
        st.markdown("---")
        st.info("💡 Contact your administrator if you've forgotten the password.")

def render_study_selection():
    render_sticky_header("study_selection")
    st.title("🔍 Study Selection")
    st.markdown("### Enter the Study ID to continue")
    index_df = load_index()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        study_input = st.text_input("Enter Study ID (e.g., GSE41998)", key="study_input", placeholder="GSE41998").strip().upper()
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        proceed_btn = st.button("Proceed →", type="primary", use_container_width=True)
    
    if proceed_btn:
        if not study_input:
            st.error("❌ Please enter a study ID.")
        elif study_input not in index_df['GSE'].values:
            st.error(f"❌ Study **{study_input}** not found in database.")
        else:
            st.session_state.selected_study = study_input
            st.session_state.page = "intro"
            st.success(f"✅ Study **{study_input}** found! Redirecting...")
            st.rerun()
    
    st.divider()
    
    # Enhanced Chatbot
    with st.expander("💬 AI Assistant - Search for Studies", expanded=True):
        st.markdown("**Ask me about available studies!**")
        st.markdown("*Try: 'search studies', 'list studies', 'GSE41998', or 'help'*")
        if "chat_history_selection" not in st.session_state:
            st.session_state.chat_history_selection = []
        user_input = st.chat_input("Ask about studies...")
        if user_input:
            st.session_state.chat_history_selection.append({"role": "user", "content": user_input})
            response_type, plot_obj, bot_message = process_chatbot_command_study_search(user_input, index_df)
            st.session_state.chat_history_selection.append({"role": "assistant", "content": bot_message})
        for message in st.session_state.chat_history_selection[-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    st.divider()
    st.markdown("### Available Studies in Database")
    if not index_df.empty:
        display_df = index_df[['GSE', 'Drug']].drop_duplicates()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No studies available in database.")

def render_intro():
    # CHANGE 3: No changes - keep as is
    render_sticky_header("intro")
    study_id = st.session_state.get("selected_study", "GSE41998")
    info = STUDY_INFO.get(study_id, STUDY_INFO["GSE41998"])
    st.markdown(f"## {info['title']}")
    st.write(info['abstract'])
    st.markdown(f"**Journal:** {info['journal']}  \n**Year:** {info['year']}  \n[View Publication]({info['link']})")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Study Selection", use_container_width=True):
            st.session_state.page = "study_selection"
            st.rerun()
    with col2:
        if st.button("View Study", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


def render_dashboard():
    # CHANGE 4: Enhanced dashboard with computations and visualization features
    study_id = st.session_state.get("selected_study", "GSE41998")
    render_sticky_header("dashboard", study_id)
    
    col_back, col_space = st.columns([1, 9])
    with col_back:
        if st.button("⬅️ Back"):
            st.session_state.page = "intro"
            st.rerun()
    
    index_df = load_index()
    if index_df.empty:
        st.warning("No studies available.")
        return

    @st.cache_data(show_spinner=True)
    def cached_dataset(gse, drug):
        return load_dataset(index_df, gse, drug)

    # Load session if exists
    if 'loaded_session' in st.session_state and st.session_state.loaded_session:
        session_data = st.session_state.loaded_session
        drug_default = session_data.get('drug')
        sample_default = session_data.get('sample', 'ALL')
        pval_default = session_data.get('pval_threshold', 0.05)
        fc_default = session_data.get('fc_threshold', 0.5)
        st.session_state.loaded_session = None
    else:
        drug_default = None
        sample_default = 'ALL'
        pval_default = 0.05
        fc_default = 0.5

    # Advanced Filters Panel
    with st.expander("🔍 Advanced Filters & Feature Selection", expanded=False):
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            study_drugs = index_df[index_df.GSE == study_id].Drug.unique()
            if drug_default and drug_default in study_drugs:
                drug_idx = list(study_drugs).index(drug_default)
            else:
                drug_idx = 0
            drug = st.selectbox("🧪 Drug", study_drugs, index=drug_idx, key="filter_drug")
        
        with filter_col2:
            clinical, expr, pathway = cached_dataset(study_id, drug)
            sample_options = ["ALL"] + list(expr.columns)
            if sample_default in sample_options:
                sample_idx = sample_options.index(sample_default)
            else:
                sample_idx = 0
            sample = st.selectbox("🔬 Sample", sample_options, index=sample_idx, key="filter_sample",
                                 help="Select a specific sample to highlight in visualizations")
        
        with filter_col3:
            response_filter = st.multiselect("📊 Response Filter", ["Sensitive", "Resistant"],
                                            default=["Sensitive", "Resistant"], key="filter_response")
        
        st.markdown("#### Visualization Feature Selection")
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            gene_options = ["None"] + sorted(list(expr.index))
            selected_gene = st.selectbox("🧬 Highlight Gene", gene_options, key="selected_gene",
                                        help="Select a gene to highlight in PCA/UMAP and heatmaps")
        
        with vis_col2:
            pathway_options = ["None"] + sorted(list(pathway.index))
            selected_pathway = st.selectbox("🔗 Highlight Pathway", pathway_options, key="selected_pathway",
                                           help="Select a pathway to highlight in pathway heatmap")
        
        st.markdown("#### Differential Expression Thresholds")
        filter_col4, filter_col5 = st.columns(2)
        with filter_col4:
            pval_threshold = st.slider("P-value Threshold", 0.0, 0.1, pval_default, 0.01,
                                      help="Filter genes by significance level", key="filter_pval")
        with filter_col5:
            fc_threshold = st.slider("Log2 Fold Change Threshold", 0.0, 3.0, fc_default, 0.1,
                                    help="Filter genes by fold change", key="filter_fc")
        st.markdown('</div>', unsafe_allow_html=True)
    
    selected = None if sample == "ALL" else sample
    highlight_gene = None if selected_gene == "None" else selected_gene
    highlight_pathway = None if selected_pathway == "None" else selected_pathway
    
    # Main Analysis Mode Tabs
    tab1, tab2 = st.tabs(["📊 Standard Analysis", "🧮 Study Computations"])
    
    with tab1:
        # Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        stats = compute_summary_stats(clinical, expr)
        de_df = compute_de(expr, clinical)
        render_metric_cards(stats)
        
        st.divider()
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Top Differential Gene:** {stats['top_gene']} (p={stats['top_gene_pval']:.2e})")
        with col_info2:
            st.info(f"**Data:** {len(expr)} genes × {len(expr.columns)} samples")
        
        st.divider()
        
        # Auto-Generated Insights
        st.markdown("### 🤖 AI-Generated Insights")
        insights = generate_auto_insights(stats, de_df, clinical, expr)
        for insight in insights:
            if insight['type'] == 'success':
                st.success(f"**{insight['title']}**: {insight['text']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['text']}")
            else:
                st.info(f"**{insight['title']}**: {insight['text']}")
        
        st.divider()
        
        # Session Management & PDF Report
        col_session, col_pdf = st.columns(2)
        
        with col_session:
            with st.expander("💾 Session Management"):
                col_save, col_load = st.columns(2)
                with col_save:
                    st.markdown("**Save Current Session**")
                    session_name = st.text_input("Session Name", placeholder="My Analysis", key="session_name_input")
                    session_notes = st.text_area("Notes (optional)", key="session_notes", height=80)
                    if st.button("Save Session", key="save_session_btn"):
                        if session_name:
                            save_session(session_name, {
                                'study_id': study_id, 'drug': drug, 'sample': sample,
                                'pval_threshold': pval_threshold, 'fc_threshold': fc_threshold,
                                'notes': session_notes
                            })
                            st.success(f"✅ Session '{session_name}' saved!")
                
                with col_load:
                    st.markdown("**Load Saved Session**")
                    sessions = get_session_list()
                    if sessions:
                        selected_session = st.selectbox("Select Session", sessions, key="session_select")
                        if st.button("Load Session", key="load_session_btn"):
                            data = load_session(selected_session)
                            if data:
                                st.session_state.loaded_session = data
                                st.success(f"✅ Loaded '{selected_session}'")
                                st.rerun()
                        if st.button("Delete Session", key="delete_session_btn"):
                            if delete_session(selected_session):
                                st.success(f"🗑️ Deleted '{selected_session}'")
                                st.rerun()
                    else:
                        st.info("No saved sessions yet")
        
        with col_pdf:
            with st.expander("📄 PDF Report"):
                if PDF_AVAILABLE:
                    st.markdown("**Generate Comprehensive Report**")
                    st.info("Includes: Summary stats, top genes, and AI insights")
                    if st.button("📄 Generate PDF Report", use_container_width=True, key="gen_pdf_btn"):
                        with st.spinner("Generating PDF..."):
                            pdf_data = generate_pdf_report(study_id, drug, stats, de_df)
                            if pdf_data:
                                st.download_button("💾 Download PDF Report", pdf_data,
                                                 f"{study_id}_{drug}_Report.pdf", "application/pdf",
                                                 key="download_pdf_btn")
                                st.success("✅ PDF Generated!")
                else:
                    st.warning("PDF generation unavailable. Install reportlab library.")
        
        st.divider()
        
        # Plot Controls
        col1, col2 = st.columns([2, 1])
        with col1:
            plot = st.selectbox("📈 Select Visualization",
                               ["All plots", "PCA", "UMAP", "Clinical Heatmap", "Gene Heatmap",
                                "Pathway Heatmap", "Volcano", "MA", "Gene Expression Table"],
                               key="plot_selector")
        with col2:
            heatmap_mode = st.radio("Heatmap Mode", ["static", "interactive"], horizontal=True, key="heatmap_mode")
        
        generate_plots = st.button("🎨 Generate Plots", type="primary", use_container_width=True, key="generate_btn")
        
        st.divider()
        
        # Main Plotting Area
        if generate_plots:
            def render_heatmap(data, title, highlight_feature=None):
                if heatmap_mode == "static":
                    st.image(static_clustermap_png(data, title), use_container_width=True)
                else:
                    fig = interactive_heatmap(data, title, selected, highlight_feature)
                    st.plotly_chart(fig, use_container_width=True)
                    col_dl1, col_dl2 = st.columns([1, 4])
                    with col_dl1:
                        if st.button("📥 Download PNG", key=f"download_{title}"):
                            img_bytes = fig_to_bytes(fig, 'png')
                            if img_bytes:
                                st.download_button("💾 Save", img_bytes, f"{title.replace(' ', '_')}.png",
                                                 "image/png", key=f"save_{title}")
            
            if plot == "All plots":
                st.markdown("### Dimensionality Reduction")
                c1, c2 = st.columns(2)
                with c1:
                    fig_pca = make_pca(expr, clinical, selected, highlight_gene)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    if st.button("📥 Download PCA", key="dl_pca"):
                        img_bytes = fig_to_bytes(fig_pca)
                        if img_bytes:
                            st.download_button("💾 Save PCA", img_bytes, "pca_plot.png", "image/png", key="save_pca")
                with c2:
                    fig_umap = make_umap(expr, clinical, selected, highlight_gene)
                    st.plotly_chart(fig_umap, use_container_width=True)
                    if st.button("📥 Download UMAP", key="dl_umap"):
                        img_bytes = fig_to_bytes(fig_umap)
                        if img_bytes:
                            st.download_button("💾 Save UMAP", img_bytes, "umap_plot.png", "image/png", key="save_umap")
                
                st.divider()
                st.markdown("### Clinical Response")
                if heatmap_mode == "static":
                    st.image(static_binary_clinical_heatmap(clinical, expr), use_container_width=True)
                else:
                    fig_clinical = interactive_binary_clinical_heatmap(clinical, expr, selected)
                    st.plotly_chart(fig_clinical, use_container_width=True)
                
                st.divider()
                st.markdown("### Gene Expression")
                render_heatmap(expr, "Gene Expression Heatmap", highlight_gene)
                
                st.divider()
                st.markdown("### Pathway Activity")
                render_heatmap(pathway, "Pathway Heatmap", highlight_pathway)
                
                st.divider()
                st.markdown("### Differential Expression")
                c3, c4 = st.columns(2)
                with c3:
                    fig_volcano = make_volcano(expr, clinical)
                    st.plotly_chart(fig_volcano, use_container_width=True)
                with c4:
                    fig_ma = make_ma(expr, clinical)
                    st.plotly_chart(fig_ma, use_container_width=True)
            
            elif plot == "PCA":
                fig = make_pca(expr, clinical, selected, highlight_gene)
                st.plotly_chart(fig, use_container_width=True)
            elif plot == "UMAP":
                fig = make_umap(expr, clinical, selected, highlight_gene)
                st.plotly_chart(fig, use_container_width=True)
            elif plot == "Clinical Heatmap":
                if heatmap_mode == "static":
                    st.image(static_binary_clinical_heatmap(clinical, expr), use_container_width=True)
                else:
                    fig = interactive_binary_clinical_heatmap(clinical, expr, selected)
                    st.plotly_chart(fig, use_container_width=True)
            elif plot == "Gene Heatmap":
                render_heatmap(expr, "Gene Expression Heatmap", highlight_gene)
            elif plot == "Pathway Heatmap":
                render_heatmap(pathway, "Pathway Heatmap", highlight_pathway)
            elif plot == "Volcano":
                fig = make_volcano(expr, clinical)
                st.plotly_chart(fig, use_container_width=True)
            elif plot == "MA":
                fig = make_ma(expr, clinical)
                st.plotly_chart(fig, use_container_width=True)
            elif plot == "Gene Expression Table":
                st.markdown("### 📋 Gene Expression Data Table")
                filtered_df = de_df[(de_df['p_value'] <= pval_threshold) & (abs(de_df['log2FC']) >= fc_threshold)].copy()
                st.info(f"Showing {len(filtered_df)} genes (filtered from {len(de_df)} total)")
                search_term = st.text_input("🔍 Search genes", placeholder="Enter gene name...", key="gene_search")
                if search_term:
                    filtered_df = filtered_df[filtered_df['gene'].str.contains(search_term, case=False, na=False)]
                display_cols = ['gene', 'log2FC', 'p_value', 'avg_expr', 'direction']
                st.dataframe(filtered_df[display_cols].sort_values('p_value'), use_container_width=True, hide_index=True, height=400)
                csv = filtered_df[display_cols].to_csv(index=False)
                st.download_button("📥 Download Table as CSV", csv, "gene_expression_table.csv", "text/csv", key="download_table")
        else:
            st.info("👆 Select your options and click 'Generate Plots' to visualize the data.")
    
    with tab2:
        # CHANGE 4: New Computation Tab
        st.markdown("### 🧮 Study-Derived Computations")
        st.info("This section provides computed metrics derived from the current study data")
        
        comp_mode = st.radio("Select Computation Mode", 
                            ["Study-Level Metrics", "Sample Profile", "Gene Profile", "Pathway Profile"],
                            horizontal=True, key="comp_mode")
        
        if comp_mode == "Study-Level Metrics":
            st.markdown("#### Overall Study Statistics")
            if st.button("🔄 Compute Study Metrics", type="primary"):
                with st.spinner("Computing metrics..."):
                    metrics = compute_study_metrics(expr, clinical, pathway)
                    
                    st.markdown('<div class="computation-panel">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Gene Expression", f"{metrics['mean_gene_expression']:.3f}")
                        st.metric("Median Gene Expression", f"{metrics['median_gene_expression']:.3f}")
                        st.metric("Gene Expression Variance", f"{metrics['gene_expression_variance']:.3f}")
                    
                    with col2:
                        st.metric("Sensitive Group Mean Expr", f"{metrics['sensitive_mean_expr']:.3f}")
                        st.metric("Resistant Group Mean Expr", f"{metrics['resistant_mean_expr']:.3f}")
                        st.metric("Expression Difference", f"{metrics['expr_difference']:.3f}")
                    
                    with col3:
                        st.metric("Mean Pathway Activity", f"{metrics['mean_pathway_activity']:.3f}")
                        st.metric("Pathway Variance", f"{metrics['pathway_variance']:.3f}")
                        st.metric("Total Genes", f"{metrics['total_genes']}")
                    
                    st.markdown("#### Differential Expression Summary")
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Significant (p<0.05)", f"{metrics['sig_genes_p005']}")
                        st.metric("Highly Significant (p<0.01)", f"{metrics['sig_genes_p001']}")
                    with col5:
                        st.metric("High FC Genes (|FC|>1)", f"{metrics['high_fc_genes']}")
                        st.metric("Very High FC (|FC|>2)", f"{metrics['very_high_fc_genes']}")
                    with col6:
                        st.metric("Higher in Sensitive", f"{metrics['genes_higher_sensitive']}")
                        st.metric("Higher in Resistant", f"{metrics['genes_higher_resistant']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Export option
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Value']
                    csv = metrics_df.to_csv()
                    st.download_button("📥 Download Metrics as CSV", csv, 
                                      f"{study_id}_{drug}_metrics.csv", "text/csv")
        
        elif comp_mode == "Sample Profile":
            st.markdown("#### Individual Sample Analysis")
            sample_to_analyze = st.selectbox("Select Sample", expr.columns, key="sample_compute")
            
            if st.button("🔄 Compute Sample Profile", type="primary"):
                with st.spinner(f"Analyzing sample {sample_to_analyze}..."):
                    profile = compute_sample_profile(sample_to_analyze, expr, clinical, pathway)
                    
                    if profile:
                        st.markdown('<div class="computation-panel">', unsafe_allow_html=True)
                        
                        st.markdown(f"### Sample: {profile['sample_id']}")
                        st.markdown(f"**Response:** {profile['response']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Gene Expression Profile")
                            st.metric("Mean Expression", f"{profile['mean_expression']:.3f}")
                            st.metric("Median Expression", f"{profile['median_expression']:.3f}")
                            st.metric("Max Expression", f"{profile['max_expression']:.3f}")
                            st.metric("Min Expression", f"{profile['min_expression']:.3f}")
                            
                            st.markdown("**Top 5 Expressed Genes:**")
                            for gene, val in profile['top_5_genes'].items():
                                st.text(f"{gene}: {val:.3f}")
                        
                        with col2:
                            st.markdown("#### Pathway Activity Profile")
                            st.metric("Mean Pathway Activity", f"{profile['mean_pathway_activity']:.3f}")
                            
                            st.markdown("**Top 5 Active Pathways:**")
                            for pw, val in profile['top_5_pathways'].items():
                                st.text(f"{pw}: {val:.3f}")
                            
                            st.markdown("#### Comparison to Group")
                            st.metric("Correlation with Group", f"{profile['vs_group_correlation']:.3f}")
                            st.metric("Deviation from Group", f"{profile['deviation_from_group']:.3f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Could not compute profile for this sample")
        
        elif comp_mode == "Gene Profile":
            st.markdown("#### Individual Gene Analysis")
            gene_to_analyze = st.selectbox("Select Gene", sorted(expr.index), key="gene_compute")
            
            if st.button("🔄 Compute Gene Profile", type="primary"):
                with st.spinner(f"Analyzing gene {gene_to_analyze}..."):
                    profile = compute_gene_profile(gene_to_analyze, expr, clinical)
                    
                    if profile:
                        st.markdown('<div class="computation-panel">', unsafe_allow_html=True)
                        
                        st.markdown(f"### Gene: {profile['gene_name']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Expression Statistics")
                            st.metric("Mean Expression", f"{profile['mean_expression']:.3f}")
                            st.metric("Median Expression", f"{profile['median_expression']:.3f}")
                            st.metric("Std Dev", f"{profile['std_expression']:.3f}")
                            st.metric("Coefficient of Variation", f"{profile['cv']:.2f}%")
                        
                        with col2:
                            st.markdown("#### Response Group Comparison")
                            st.metric("Sensitive Mean", f"{profile['sensitive_mean']:.3f}")
                            st.metric("Resistant Mean", f"{profile['resistant_mean']:.3f}")
                            st.metric("Fold Change (Log2)", f"{profile['fold_change']:.3f}")
                            st.metric("P-value", f"{profile['p_value']:.2e}")
                            if profile['significant']:
                                st.success("✅ Statistically Significant (p<0.05)")
                            else:
                                st.warning("⚠️ Not Significant (p≥0.05)")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Could not compute profile for this gene")
        
        elif comp_mode == "Pathway Profile":
            st.markdown("#### Individual Pathway Analysis")
            pathway_to_analyze = st.selectbox("Select Pathway", sorted(pathway.index), key="pathway_compute")
            
            if st.button("🔄 Compute Pathway Profile", type="primary"):
                with st.spinner(f"Analyzing pathway {pathway_to_analyze}..."):
                    profile = compute_pathway_profile(pathway_to_analyze, pathway, clinical)
                    
                    if profile:
                        st.markdown('<div class="computation-panel">', unsafe_allow_html=True)
                        
                        st.markdown(f"### Pathway: {profile['pathway_name']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Activity Statistics")
                            st.metric("Mean Activity", f"{profile['mean_activity']:.3f}")
                            st.metric("Median Activity", f"{profile['median_activity']:.3f}")
                            st.metric("Std Dev", f"{profile['std_activity']:.3f}")
                        
                        with col2:
                            st.markdown("#### Response Group Comparison")
                            st.metric("Sensitive Mean", f"{profile['sensitive_mean']:.3f}")
                            st.metric("Resistant Mean", f"{profile['resistant_mean']:.3f}")
                            st.metric("Activity Difference", f"{profile['activity_difference']:.3f}")
                            st.metric("P-value", f"{profile['p_value']:.2e}")
                            if profile['significant']:
                                st.success("✅ Statistically Significant (p<0.05)")
                            else:
                                st.warning("⚠️ Not Significant (p≥0.05)")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Could not compute profile for this pathway")


# ==================================================
# MAIN APPLICATION
# ==================================================
def main():
    st.set_page_config(page_title="Clinical-Omics Explorer - Refined Edition", 
                       layout="wide", initial_sidebar_state="auto")
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "selected_study" not in st.session_state:
        st.session_state.selected_study = None
    if 'theme' not in st.session_state:
        st.session_state.theme = load_theme()
    
    # CHANGE 2: Simplified sidebar without version/features
    if st.session_state.authenticated:
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            current_theme = st.session_state.theme
            theme_label = "🌙 Dark Mode" if current_theme == 'light' else "☀️ Light Mode"
            
            if st.button(theme_label, use_container_width=True):
                new_theme = 'dark' if current_theme == 'light' else 'light'
                st.session_state.theme = new_theme
                save_theme(new_theme)
                st.rerun()
            
            st.markdown("---")
            st.markdown("**Current Theme:**")
            st.info(f"{'🌙 Dark' if current_theme == 'dark' else '☀️ Light'}")
    
    # Apply theme
    apply_custom_css(st.session_state.theme)
    
    # Route to appropriate page
    if not st.session_state.authenticated:
        render_login()
    else:
        if st.session_state.page == "study_selection":
            render_study_selection()
        elif st.session_state.page == "intro":
            render_intro()
        elif st.session_state.page == "dashboard":
            render_dashboard()
        else:
            render_study_selection()

if __name__ == "__main__":
    main()