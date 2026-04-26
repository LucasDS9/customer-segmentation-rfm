import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="RFM Dashboard", layout="wide", page_icon="📊")
plt.rcParams["axes.grid"] = False

st.markdown("""
<style>
    body, .stApp { background-color: #ffffff !important; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; background-color: #ffffff; }
    .metric-card {
        background: #f5eeff;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #c89df0;
    }
    .metric-label { font-size: 11px; color: #7b2fbe; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #5a1a9b; }
    .metric-card.vip { background: #ede0ff; border-color: #a855f7; }
    .metric-card.vip .metric-value { color: #6b21a8; }
    .metric-card.regular { background: #f0e8ff; border-color: #b07cf0; }
    .metric-card.regular .metric-value { color: #7c3aed; }
    .metric-card.risk { background: #ede0ff; border-color: #9b59b6; }
    .metric-card.risk .metric-value { color: #5b21b6; }
    .section-header {
        font-size: 11px;
        font-weight: 600;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.75rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 6px;
    }
    .rfm-box {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #9b59b6;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 1.5rem;
    }
    .insight-box {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 1rem;
    }
    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        color: #3b0764;
        margin-bottom: 0.4rem;
    }
    .dashboard-intro {
        font-size: 14px;
        color: #555;
        line-height: 1.65;
        margin-bottom: 1.5rem;
        max-width: 960px;
    }
    .dist-box {
        background: #f8f4ff;
        border: 1px solid #d8c4f5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .dist-box-title {
        font-size: 10px;
        font-weight: 700;
        color: #7b2fbe;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }
    .dist-row {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        gap: 8px;
    }
    .dist-label {
        font-size: 11px;
        color: #444;
        min-width: 90px;
        font-weight: 500;
    }
    .dist-bar-wrap {
        flex: 1;
        background: #e9d9ff;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .dist-bar {
        height: 10px;
        border-radius: 4px;
        background: linear-gradient(90deg, #9b59b6, #7b2fbe);
    }
    .dist-pct {
        font-size: 11px;
        font-weight: 600;
        color: #5a1a9b;
        min-width: 42px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-title">📊 RFM Dashboard — Segmentação de Clientes</div>', unsafe_allow_html=True)
st.markdown("""
<div class="dashboard-intro">
A segmentação de clientes é uma das estratégias mais importantes para empresas de e-commerce,
permitindo compreender diferentes perfis de consumo, otimizar campanhas de marketing, melhorar
retenção e aumentar o valor do cliente ao longo do tempo. Neste projeto, foi aplicada a metodologia
<strong>RFM (Recency, Frequency, Monetary)</strong> combinada com técnicas de clusterização não
supervisionada para segmentar clientes de um e-commerce real, utilizando o dataset
<em>Online Retail</em>.
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    rfm    = pd.read_csv("data/rfm.csv")
    scaled = pd.read_csv("data/rfm_scaled.csv")
    pca_df = pd.read_csv("data/rfm_pca.csv")

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(scaled)
    rfm["cluster_raw"] = km.labels_
    mapping = {0: "Clientes de Risco", 1: "Clientes Regulares", 2: "VIP"}
    rfm["cluster"] = rfm["cluster_raw"].map(mapping)
    rfm["PC1"] = pca_df["PC1"]
    rfm["PC2"] = pca_df["PC2"]

    dbscan = DBSCAN(eps=0.8, min_samples=5)
    rfm["cluster_db"] = dbscan.fit_predict(scaled)

    mask         = rfm["cluster_db"] != -1
    rfm_clean    = rfm[mask].copy()
    scaled_clean = scaled[mask.values].copy()

    km_clean = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_clean.fit(scaled_clean)
    rfm_clean["cluster_raw"] = km_clean.labels_
    rfm_clean["cluster"]     = rfm_clean["cluster_raw"].map(mapping)

    pca_model = PCA(n_components=2)
    rfm_pca_clean = pd.DataFrame(
        pca_model.fit_transform(scaled_clean),
        columns=["PC1", "PC2"],
        index=rfm_clean.index,
    )
    rfm_pca_clean["cluster"] = rfm_clean["cluster"].values

    rfm_cluster_summary = (
        rfm_clean
        .groupby("cluster")
        .agg(
            customers=("CustomerID", "count"),
            recency_mean=("Recency", "mean"),
            recency_median=("Recency", "median"),
            frequency_mean=("Frequency", "mean"),
            frequency_median=("Frequency", "median"),
            monetary_mean=("Monetary", "mean"),
            monetary_median=("Monetary", "median"),
        )
    )
    rfm_cluster_summary["R_rank"] = rfm_cluster_summary["recency_median"].rank(ascending=True)
    rfm_cluster_summary["F_rank"] = rfm_cluster_summary["frequency_median"].rank(ascending=False)
    rfm_cluster_summary["M_rank"] = rfm_cluster_summary["monetary_median"].rank(ascending=False)
    rfm_cluster_summary = rfm_cluster_summary.sort_values(by=["R_rank", "F_rank", "M_rank"])

    return rfm, rfm_clean, rfm_pca_clean, rfm_cluster_summary


@st.cache_data
def build_rfm_chart_data(rfm: pd.DataFrame) -> pd.DataFrame:
    def winsorize_s(series, lower=0.01, upper=0.99):
        return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))

    rfm_w = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_w[col] = winsorize_s(rfm[col])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_w[["Recency", "Frequency", "Monetary"]])

    db        = DBSCAN(eps=1.5, min_samples=10)
    labels_db = db.fit_predict(X_scaled)

    rfm_chart = rfm_w[labels_db != -1].copy()

    label_map = {2: "VIP", 1: "Clientes Regulares", 0: "Clientes de Risco"}
    rfm_chart["cluster_nome"] = rfm_chart["cluster_raw"].map(label_map)

    return rfm_chart


rfm, rfm_clean, rfm_pca_clean, rfm_cluster_summary = load_data()

cmap       = plt.cm.magma
COLOR_VIP  = cmap(0.95)
COLOR_REG  = cmap(0.50)
COLOR_RISK = cmap(0.15)

PALETTE = {
    "VIP":                COLOR_VIP,
    "Clientes Regulares": COLOR_REG,
    "Clientes de Risco":  COLOR_RISK,
}

total    = len(rfm_clean)
n_vip    = (rfm_clean["cluster"] == "VIP").sum()
n_reg    = (rfm_clean["cluster"] == "Clientes Regulares").sum()
n_risk   = (rfm_clean["cluster"] == "Clientes de Risco").sum()
rec_mean = round(rfm_clean["Recency"].mean(), 1)
frq_mean = round(rfm_clean["Frequency"].mean(), 1)
mon_mean = round(rfm_clean["Monetary"].mean(), 2)

# Helper: format number with dots as thousand separator (PT-BR style)
def fmt_pt(n):
    return f"{n:,}".replace(",", ".")

st.markdown("""
<div class="rfm-box">
<strong style="font-size:15px;">RFM — Segmentação de Clientes</strong>
<div style="display:flex; gap:2rem; margin-top:10px; flex-wrap:wrap;">
  <div><b>R — Recency (Recência)</b><br><span style="font-size:13px;color:#666;">Há quanto tempo o cliente fez a última compra. Clientes que compraram recentemente têm menor probabilidade de deixar de ser clientes.</span></div>
  <div><b>F — Frequency (Frequência)</b><br><span style="font-size:13px;color:#666;">Quantas vezes o cliente comprou. Clientes frequentes tendem a ter maior engajamento e fidelidade.</span></div>
  <div><b>M — Monetary (Valor Monetário)</b><br><span style="font-size:13px;color:#666;">Quanto dinheiro o cliente gastou no total. Ajuda a identificar clientes que geram mais receita.</span></div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">Visão geral da base</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total de clientes</div><div class="metric-value">{fmt_pt(total)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card vip"><div class="metric-label">Clientes VIP</div><div class="metric-value">{fmt_pt(n_vip)}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card regular"><div class="metric-label">Clientes Regulares</div><div class="metric-value">{fmt_pt(n_reg)}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card risk"><div class="metric-label">Clientes de Risco</div><div class="metric-value">{fmt_pt(n_risk)}</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Recência média</div><div class="metric-value">{rec_mean}d</div></div>', unsafe_allow_html=True)
with c6:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Freq. média</div><div class="metric-value">{frq_mean}x</div></div>', unsafe_allow_html=True)
with c7:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Receita média</div><div class="metric-value">{fmt_pt(int(mon_mean))}</div></div>', unsafe_allow_html=True)

# ── Distribuições (rfm completo, com winsorize) ──
st.markdown('<div class="section-header">Distribuição RFM (winsorizado)</div>', unsafe_allow_html=True)

def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

rfm_w = rfm.copy()
for col in ["Recency", "Frequency", "Monetary"]:
    rfm_w[col] = winsorize(rfm[col])

# Distribution data for the boxes
FREQ_DIST = [
    ("0 – 5",   75.36),
    ("5 – 10",  16.69),
    ("10 – 15+", 100 - 75.36 - 16.69),
]
RECENCY_DIST = [
    ("0 – 50",    48.88),
    ("50 – 100",  19.54),
    ("100 – 150+", 100 - 48.88 - 19.54),
]
MONETARY_DIST = [
    ("0 – 2.000",      79.63),
    ("2.000 – 4.000",  12.04),
    ("4.000 – 6.000+", 100 - 79.63 - 12.04),
]

def dist_box_html(title, rows):
    html = f'<div class="dist-box"><div class="dist-box-title">Distribuição — {title}</div>'
    for label, pct in rows:
        bar_w = int(pct)
        html += f"""
        <div class="dist-row">
            <span class="dist-label">{label}</span>
            <div class="dist-bar-wrap">
                <div class="dist-bar" style="width:{bar_w}%;"></div>
            </div>
            <span class="dist-pct">{pct:.2f}%</span>
        </div>"""
    html += '</div>'
    return html

h1, h2, h3 = st.columns(3)
for col_, ax_col, color_, label_, dist_title, dist_rows in [
    ("Frequency", h1, cmap(0.95), "Frequência",  "Frequency",  FREQ_DIST),
    ("Recency",   h2, cmap(0.50), "Recência",    "Recency",    RECENCY_DIST),
    ("Monetary",  h3, cmap(0.15), "Monetário",   "Monetary",   MONETARY_DIST),
]:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.hist(rfm_w[col_], bins=15, color=color_, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Distribuição de {label_}", fontsize=11, pad=8, color="#333")
    ax.set_xlabel(col_, fontsize=9, color="#555")
    ax.set_ylabel("Quantidade", fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    with ax_col:
        st.pyplot(fig, use_container_width=True)
        st.markdown(dist_box_html(dist_title, dist_rows), unsafe_allow_html=True)
    plt.close()

st.markdown("""
<div class="insight-box">
  <p>Esses três histogramas mostram a distribuição das métricas de RFM (Recency, Frequency, Monetary):</p>
  <p><strong>Recency (Recência):</strong> Forte concentração em valores baixos → muitos clientes compraram recentemente. A cauda longa indica alguns clientes inativos há bastante tempo.</p>
  <p><strong>Frequency (Frequência):</strong> A maioria compra poucas vezes (1–5), com poucos clientes muito frequentes. Distribuição bem assimétrica à direita indicando que uma minoria compra frequentemente.</p>
  <p><strong>Monetary (Valor gasto):</strong> Grande parte dos clientes gasta pouco, enquanto poucos gastam valores muito altos (outliers). Também é fortemente enviesada à direita indicando uma concentração baixa de monetary pela maioria.</p>
</div>
""", unsafe_allow_html=True)

# ── Scatter plots ──
st.markdown('<div class="section-header">Clusters — visualização</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)

def scatter_plot(x_col, y_col, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for label, color in PALETTE.items():
        mask = rfm_clean["cluster"] == label
        ax.scatter(
            rfm_clean.loc[mask, x_col],
            rfm_clean.loc[mask, y_col],
            c=[color], label=label, alpha=0.6, s=12, linewidths=0
        )
    ax.set_title(title, fontsize=11, pad=8, color="#333")
    ax.set_xlabel(x_label, fontsize=9, color="#555")
    ax.set_ylabel(y_label, fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    patches = [mpatches.Patch(color=c, label=l) for l, c in PALETTE.items()]
    ax.legend(handles=patches, fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout()
    return fig

with s1:
    fig = scatter_plot("Recency", "Frequency", "Recency vs Frequency", "Recency", "Frequency")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with s2:
    fig = scatter_plot("Monetary", "Recency", "Monetary vs Recency", "Monetary", "Recency")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with s3:
    palette_pca = {
        "VIP":                cmap(0.95),
        "Clientes Regulares": cmap(0.15),
        "Clientes de Risco":  cmap(0.4),
    }
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    sns.scatterplot(
        data=rfm_pca_clean,
        x="PC1", y="PC2",
        hue="cluster",
        palette=palette_pca,
        alpha=0.7, ax=ax, s=20,
    )
    ax.set_title("Clusters via PCA (sem outliers)", fontsize=11, pad=8, color="#333")
    ax.set_xlabel("Componente Principal 1", fontsize=9, color="#555")
    ax.set_ylabel("Componente Principal 2", fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.legend(title="Segmento", fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("""
<div class="insight-box">
  <p>
    <strong>Os clientes VIP</strong> apresentam alta frequência e baixa recência, ou seja, compram muitas vezes e ficam pouco tempo sem realizar novas compras.
    <strong>Os clientes de risco</strong> têm baixa frequência e recência alta, comprando poucas vezes e demorando para retornar.
    <strong>Os clientes regulares</strong> exibem variedade na frequência e recência médias.
    Ao analisar exclusivamente a variável <strong>Monetary</strong>, observa-se que os maiores valores de gasto estão concentrados no <strong>Cluster VIP</strong>, reforçando seu perfil de clientes de alto valor para o negócio.
    Os grupos de clientes regulares e de risco compartilham uma faixa semelhante de valores monetários na maior parte dos registros; no entanto, os <strong>Clientes Regulares se destacam por conter indivíduos com valores de gasto superiores</strong>, indicando a presença de clientes com potencial de crescimento e maior ticket médio dentro desse grupo.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">Distribuição RFM por cluster (sem outliers)</div>', unsafe_allow_html=True)

rfm_chart = build_rfm_chart_data(rfm)

palette_chart = {
    "VIP":                cmap(0.95),
    "Clientes Regulares": cmap(0.15),
    "Clientes de Risco":  cmap(0.4),
}

# 🔧 tamanho menor + mais qualidade
import numpy as np

fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.8), dpi=150)
fig.patch.set_facecolor("white")

# KDE — Recency
sns.kdeplot(
    data=rfm_chart, x="Recency", hue="cluster_nome",
    palette=palette_chart, fill=True,
    common_norm=False, alpha=0.5, linewidth=1.2, ax=ax[0]
)
ax[0].set_title("Recency", fontsize=9)

# KDE — Frequency
sns.kdeplot(
    data=rfm_chart, x="Frequency", hue="cluster_nome",
    palette=palette_chart, fill=True,
    common_norm=False, alpha=0.5, linewidth=1.2, ax=ax[1]
)
ax[1].set_title("Frequency", fontsize=9)

# HISTOGRAMA — Monetary
sns.histplot(
    data=rfm_chart,
    x="Monetary",
    hue="cluster_nome",
    palette=palette_chart,
    bins=30,
    alpha=0.5,
    element="step",     # mais clean
    stat="density",
    common_norm=False,
    ax=ax[2]
)

ax[2].set_title("Monetary", fontsize=9)
ax[2].tick_params(axis="x", labelsize=7)

# estilo clean (igual aos outros)
for a in ax:
    a.set_facecolor("white")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(False)
    a.tick_params(labelsize=7)

# legenda única e pequena
handles, labels_leg = ax[0].get_legend_handles_labels()
fig.legend(
    handles, labels_leg,
    loc="upper center", ncol=3,
    frameon=False, fontsize=6
)

# remove legendas duplicadas
for a in ax:
    if a.get_legend():
        a.get_legend().remove()

plt.tight_layout(rect=[0, 0, 1, 0.9])
st.pyplot(fig)
plt.close()

st.markdown("""
<div class="insight-box">
  <p>
    <strong>Recency:</strong> Clientes VIP (branco) concentrados em valores baixos (~0–30 dias), indicando compras recentes;
    Clientes de Risco (roxo claro) no meio (~20–100 dias); Regulares (roxo escuro) com recência alta (~150–400 dias), ou seja, estão há muito tempo sem comprar.
  </p>
  <p>
    <strong>Frequency:</strong> VIP (branco) com frequência alta (~25–35 compras); Clientes de Risco (roxo claro) baixos (~1–5);
    Regulares (roxo escuro) muito baixos (~1–3), mostrando pouco engajamento.
  </p>
  <p>
    <strong>Monetary:</strong> VIP (branco) com valores muito altos (podendo chegar perto de ~19k);
    Clientes de Risco (roxo claro) com valores médios (~500–3000); Regulares (roxo escuro) com baixos gastos (~100–1000).
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">Heatmap RFM por cluster</div>', unsafe_allow_html=True)

heat_data = rfm_cluster_summary[["recency_mean", "frequency_mean", "monetary_mean"]]

_, heat_col, _ = st.columns([1, 2, 1])
with heat_col:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    sns.heatmap(
        heat_data,
        annot=True, fmt=".1f", cmap="magma",
        linewidths=0.5, linecolor="#f0f0f0",
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("RFM Cluster Heatmap", fontsize=11, pad=10, color="#333")
    ax.tick_params(labelsize=9, colors="#555")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("""
<div class="insight-box">
  <p>
    <strong>VIP:</strong> recency ≈ 14 dias, frequency ≈ 15,5, monetary ≈ 6097 → compram recentemente, com alta frequência e alto gasto, sendo os clientes mais valiosos.
  </p>
  <p>
    <strong>Clientes de Risco:</strong> recency ≈ 46 dias, frequency ≈ 3,2, monetary ≈ 1151 → já estão há mais tempo sem comprar, com baixa frequência, indicando risco de churn.
  </p>
  <p>
    <strong>Clientes Regulares:</strong> recency ≈ 249 dias, frequency ≈ 1,5, monetary ≈ 473 → quase inativos, compram pouco e geram baixo valor.
  </p>
</div>
""", unsafe_allow_html=True)