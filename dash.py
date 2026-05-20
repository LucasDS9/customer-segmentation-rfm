import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
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
        padding: 14px 18px;
        border: 1px solid #c89df0;
        min-height: 72px;
        box-sizing: border-box;
    }
    .metric-label {
        font-size: 9.5px;
        color: #7b2fbe;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .metric-value { font-size: 26px; font-weight: 600; color: #5a1a9b; }
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
        font-size: 10px; font-weight: 700; color: #7b2fbe;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px;
    }
    .dist-row { display: flex; align-items: center; margin-bottom: 5px; gap: 8px; }
    .dist-label { font-size: 11px; color: #444; min-width: 90px; font-weight: 500; }
    .dist-bar-wrap { flex: 1; background: #e9d9ff; border-radius: 4px; height: 10px; overflow: hidden; }
    .dist-bar { height: 10px; border-radius: 4px; background: linear-gradient(90deg, #9b59b6, #7b2fbe); }
    .dist-pct { font-size: 11px; font-weight: 600; color: #5a1a9b; min-width: 42px; text-align: right; }

    /* ── Tabela executiva ── */
    .exec-header {
        display: flex; align-items: center; gap: 10px;
        background: linear-gradient(90deg, #2d1b4e 0%, #4a2080 100%);
        border-radius: 10px 10px 0 0;
        padding: 14px 22px; margin-top: 2rem;
    }
    .exec-header-icon  { font-size: 22px; }
    .exec-header-title { font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: .01em; }
    .exec-header-sub   { font-size: 11px; color: #c9a8f5; margin-top: 1px; }
    .exec-kpi-row {
        background: #1e0f35; padding: 16px 22px 14px 22px; display: flex; gap: 0;
    }
    .exec-kpi-card            { flex: 1; padding: 8px 16px; border-right: 1px solid #3d2060; }
    .exec-kpi-card:last-child { border-right: none; }
    .exec-kpi-label { font-size: 9px; color: #b39ddb; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
    .exec-kpi-value { font-size: 22px; font-weight: 700; color: #f3e5ff; }
    .exec-metrics-row {
        background: #fdf0f8; border: 1px solid #f0c8e8; border-top: none;
        padding: 14px 22px; display: flex; gap: 12px;
    }
    .exec-chip       { flex: 1; background: #fff; border: 1px solid #e8b4d8; border-radius: 8px; padding: 10px 14px; text-align: center; }
    .exec-chip-label { font-size: 9px; color: #b05090; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
    .exec-chip-value { font-size: 18px; font-weight: 700; color: #7b1f5a; }
    .exec-chip-value.neg { color: #c0392b; }
    .exec-chip-value.pos { color: #1a7a4a; }
    .exec-list-wrapper {
        background: #ffffff; border: 1px solid #e0d0f0;
        border-top: none; border-radius: 0 0 10px 10px;
        padding: 16px 22px 20px 22px;
    }
    .exec-list-title { font-size: 11px; font-weight: 700; color: #7b2fbe; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px; }
    .exec-record-wrap { background: #fdf8ff; border: 1.5px solid #c89df0; border-radius: 10px; padding: 18px 24px; margin-top: 14px; }
    .exec-record-title { font-size: 13px; font-weight: 700; color: #4a1080; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
    .exec-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .exec-cell { background: #fff; border: 1px solid #e0d0f0; border-radius: 7px; padding: 10px 14px; }
    .exec-cell-label { font-size: 9px; color: #9b59b6; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 3px; }
    .exec-cell-value { font-size: 15px; font-weight: 600; color: #2d1b4e; }
    .exec-cell-value.neg { color: #c0392b; }
    .exec-cell-value.pos { color: #1a7a4a; }
    .exec-badge {
        display: inline-block; background: #f3e5ff; color: #6b21a8;
        border: 1px solid #c89df0; border-radius: 20px;
        font-size: 11px; font-weight: 600; padding: 2px 12px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TÍTULO E INTRO
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="dashboard-title">📊 RFM Dashboard — Segmentação de Clientes</div>', unsafe_allow_html=True)
st.markdown("""
<div class="dashboard-intro">
A segmentação de clientes é uma das estratégias mais importantes para empresas de e-commerce,
permitindo compreender diferentes perfis de consumo, otimizar campanhas de marketing, melhorar
retenção e aumentar o valor do cliente ao longo do tempo. Neste projeto, foi aplicada a metodologia
<strong>RFM (Recency, Frequency, Monetary)</strong> combinada com técnicas de clusterização não
supervisionada para segmentar clientes de um e-commerce real, utilizando o dataset <em>Online Retail</em>.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    rfm    = pd.read_csv("data/rfm.csv")
    scaled = pd.read_csv("data/rfm_scaled.csv")
    pca_df = pd.read_csv("data/rfm_pca.csv")

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(scaled)
    rfm["cluster_raw"] = km.labels_

    mapping = {0: "Clientes regulares", 1: "Clientes ocasionais", 2: "Clientes engajados"}
    rfm["cluster"] = rfm["cluster_raw"].map(mapping)
    rfm["PC1"] = pca_df["PC1"]
    rfm["PC2"] = pca_df["PC2"]

    pca_model = PCA(n_components=2)
    rfm_pca = pd.DataFrame(
        pca_model.fit_transform(scaled),
        columns=["PC1", "PC2"],
        index=rfm.index,
    )
    rfm_pca["cluster"] = rfm["cluster"].values

    rfm_cluster_summary = (
        rfm.groupby("cluster").agg(
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

    return rfm, rfm_pca, rfm_cluster_summary


@st.cache_data
def load_rfmplus():
    return pd.read_csv("data/rfmplus.csv")


rfm, rfm_pca, rfm_cluster_summary = load_data()
rfmplus = load_rfmplus()

cmap             = plt.cm.magma

# ── CORREÇÃO: engajados → roxo escuro (0.15), regulares → amarelo (0.95) ──
COLOR_ENGAJADOS  = cmap(0.15)   # roxo escuro  ← corrigido
COLOR_REG        = cmap(0.95)   # amarelo claro ← corrigido
COLOR_OCASIONAIS = cmap(0.50)   # rosa/magenta  (inalterado)

PALETTE = {
    "Clientes engajados":  COLOR_ENGAJADOS,
    "Clientes regulares":  COLOR_REG,
    "Clientes ocasionais": COLOR_OCASIONAIS,
}

# Contagens direto do rfmplus (fonte da verdade)
total  = len(rfmplus)
n_eng  = (rfmplus["Grupo"] == "Engajado").sum()
n_reg  = (rfmplus["Grupo"] == "Regular").sum()
n_ocas = (rfmplus["Grupo"] == "Ocasional").sum()

rec_mean = round(rfmplus["Recency"].mean(), 1)
frq_mean = round(rfmplus["Frequency"].mean(), 1)
mon_mean = round(rfmplus["Monetary"].mean(), 2)


def fmt_pt(n):
    return f"{n:,}".replace(",", ".")


# ══════════════════════════════════════════════════════════════
#  BOX RFM
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="rfm-box">
<strong style="font-size:15px;">RFM — Segmentação de Clientes</strong>
<div style="display:flex; gap:2rem; margin-top:10px; flex-wrap:wrap;">
  <div><b>R — Recency (Recência)</b><br><span style="font-size:13px;color:#666;">Há quanto tempo o cliente fez a última compra.</span></div>
  <div><b>F — Frequency (Frequência)</b><br><span style="font-size:13px;color:#666;">Quantas vezes o cliente comprou.</span></div>
  <div><b>M — Monetary (Valor Monetário)</b><br><span style="font-size:13px;color:#666;">Quanto dinheiro o cliente gastou no total.</span></div>
</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  KPIs — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Visão geral da base</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total de clientes</div><div class="metric-value">{fmt_pt(total)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Clientes engajados</div><div class="metric-value">{fmt_pt(n_eng)}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Clientes regulares</div><div class="metric-value">{fmt_pt(n_reg)}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Clientes ocasionais</div><div class="metric-value">{fmt_pt(n_ocas)}</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Recência média</div><div class="metric-value">{rec_mean}d</div></div>', unsafe_allow_html=True)
with c6:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Freq. média</div><div class="metric-value">{frq_mean}x</div></div>', unsafe_allow_html=True)
with c7:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Receita média</div><div class="metric-value">{fmt_pt(int(mon_mean))}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DISTRIBUIÇÕES RFM
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Distribuição RFM (winsorizado)</div>', unsafe_allow_html=True)


def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))


rfm_w = rfm.copy()
for col in ["Recency", "Frequency", "Monetary"]:
    rfm_w[col] = winsorize(rfm[col])

FREQ_DIST     = [("0 – 5", 75.36), ("5 – 10", 16.69), ("10 – 15+", 100 - 75.36 - 16.69)]
RECENCY_DIST  = [("0 – 50", 48.88), ("50 – 100", 19.54), ("100 – 150+", 100 - 48.88 - 19.54)]
MONETARY_DIST = [("0 – 2.000", 79.63), ("2.000 – 4.000", 12.04), ("4.000 – 6.000+", 100 - 79.63 - 12.04)]


def dist_box_html(title, rows):
    html = f'<div class="dist-box"><div class="dist-box-title">Distribuição — {title}</div>'
    for label, pct in rows:
        html += f"""
        <div class="dist-row">
            <span class="dist-label">{label}</span>
            <div class="dist-bar-wrap"><div class="dist-bar" style="width:{int(pct)}%;"></div></div>
            <span class="dist-pct">{pct:.2f}%</span>
        </div>"""
    return html + '</div>'


h1, h2, h3 = st.columns(3)
for col_, ax_col, color_, label_, dist_title, dist_rows in [
    ("Frequency", h1, cmap(0.95), "Frequência", "Frequency", FREQ_DIST),
    ("Recency",   h2, cmap(0.50), "Recência",   "Recency",   RECENCY_DIST),
    ("Monetary",  h3, cmap(0.15), "Monetário",  "Monetary",  MONETARY_DIST),
]:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.hist(rfm_w[col_], bins=15, color=color_, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Distribuição de {label_}", fontsize=11, pad=8, color="#333")
    ax.set_xlabel(col_, fontsize=9, color="#555"); ax.set_ylabel("Quantidade", fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(False); plt.tight_layout()
    with ax_col:
        st.pyplot(fig, use_container_width=True)
        st.markdown(dist_box_html(dist_title, dist_rows), unsafe_allow_html=True)
    plt.close()

st.markdown("""
<div class="insight-box">
  <p>Esses três histogramas mostram a distribuição das métricas de RFM:</p>
  <p><strong>Recency:</strong> Forte concentração em valores baixos → muitos clientes compraram recentemente. A cauda longa indica alguns inativos há bastante tempo.</p>
  <p><strong>Frequency:</strong> A maioria compra poucas vezes (1–5). Distribuição assimétrica à direita.</p>
  <p><strong>Monetary:</strong> Grande parte gasta pouco, poucos gastam valores altos. Também fortemente enviesada à direita.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SCATTER PLOTS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Clusters — visualização</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)


def scatter_plot(x_col, y_col, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for label, color in PALETTE.items():
        mask = rfm["cluster"] == label
        ax.scatter(rfm.loc[mask, x_col], rfm.loc[mask, y_col],
                   c=[color], label=label, alpha=0.6, s=12, linewidths=0)
    ax.set_title(title, fontsize=11, pad=8, color="#333")
    ax.set_xlabel(x_label, fontsize=9, color="#555"); ax.set_ylabel(y_label, fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(False)
    patches = [mpatches.Patch(color=c, label=l) for l, c in PALETTE.items()]
    ax.legend(handles=patches, fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout()
    return fig


with s1:
    fig = scatter_plot("Recency", "Frequency", "Recency vs Frequency", "Recency", "Frequency")
    st.pyplot(fig, use_container_width=True); plt.close()
with s2:
    fig = scatter_plot("Monetary", "Recency", "Monetary vs Recency", "Monetary", "Recency")
    st.pyplot(fig, use_container_width=True); plt.close()
with s3:
    # ── CORREÇÃO: palette_pca usa o PALETTE global já corrigido ──
    palette_pca = {
        "Clientes engajados":  COLOR_ENGAJADOS,   # roxo escuro
        "Clientes regulares":  COLOR_REG,         # amarelo
        "Clientes ocasionais": COLOR_OCASIONAIS,  # magenta
    }
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    sns.scatterplot(data=rfm_pca, x="PC1", y="PC2", hue="cluster", palette=palette_pca, alpha=0.7, ax=ax, s=20)
    ax.set_title("Clusters via PCA", fontsize=11, pad=8, color="#333")
    ax.set_xlabel("Componente Principal 1", fontsize=9, color="#555")
    ax.set_ylabel("Componente Principal 2", fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(False)
    ax.legend(title="Segmento", fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════
#  FUNÇÃO AUXILIAR — TABELA EXECUTIVA
# ══════════════════════════════════════════════════════════════
def render_tabela_executiva(grupo_key: str, titulo: str, icone: str, selectbox_key: str):
    df = rfmplus[rfmplus["Grupo"] == grupo_key].copy()

    qtd         = len(df)
    rec_med     = int(df["Recency"].median())
    frq_med     = int(df["Frequency"].median())
    mon_med     = df["Monetary"].median()
    cac_mean    = df["CAC"].mean()
    roi_med     = df["ROI"].median() * 100
    lucro_mean  = df["Lucro_Cliente"].mean()
    roi_pos_n   = (df["ROI"] > 0).sum()
    roi_pos_pct = (df["ROI"] > 0).mean() * 100
    roi_neg_n   = (df["ROI"] < 0).sum()
    roi_neg_pct = (df["ROI"] < 0).mean() * 100

    qtd_fmt     = f"{qtd:,}".replace(",", ".")
    mon_med_fmt = f"R$ {mon_med:,.2f}".replace(",", ".")
    cac_fmt     = f"R$ {cac_mean:,.2f}".replace(",", ".")
    roi_fmt     = f"{roi_med:+.2f}%"
    roi_cls     = "neg" if roi_med < 0 else "pos"
    lucro_fmt   = f"R$ {lucro_mean:+,.2f}".replace(",", ".")
    lucro_cls   = "neg" if lucro_mean < 0 else "pos"

    st.markdown(f"""
    <div class="exec-header">
      <span class="exec-header-icon">{icone}</span>
      <div>
        <div class="exec-header-title">{titulo}</div>
        <div class="exec-header-sub">Visão individual de clientes por segmento</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="exec-kpi-row">
      <div class="exec-kpi-card"><div class="exec-kpi-label">Quantidade</div><div class="exec-kpi-value">{qtd_fmt}</div></div>
      <div class="exec-kpi-card"><div class="exec-kpi-label">Recência Mediana</div><div class="exec-kpi-value">{rec_med}d</div></div>
      <div class="exec-kpi-card"><div class="exec-kpi-label">Frequência Mediana</div><div class="exec-kpi-value">{frq_med}x</div></div>
      <div class="exec-kpi-card"><div class="exec-kpi-label">Monetário Mediano</div><div class="exec-kpi-value">{mon_med_fmt}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="exec-metrics-row">
      <div class="exec-chip"><div class="exec-chip-label">CAC Médio</div><div class="exec-chip-value">{cac_fmt}</div></div>
      <div class="exec-chip"><div class="exec-chip-label">ROI Mediano</div><div class="exec-chip-value {roi_cls}">{roi_fmt}</div></div>
      <div class="exec-chip"><div class="exec-chip-label">Lucro Médio</div><div class="exec-chip-value {lucro_cls}">{lucro_fmt}</div></div>
      <div class="exec-chip"><div class="exec-chip-label">ROI Positivo</div><div class="exec-chip-value pos">{roi_pos_n} ({roi_pos_pct:.1f}%)</div></div>
      <div class="exec-chip"><div class="exec-chip-label">ROI Negativo</div><div class="exec-chip-value neg">{roi_neg_n} ({roi_neg_pct:.1f}%)</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="exec-list-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="exec-list-title">🔍 Selecione um cliente para ver o registro completo</div>', unsafe_allow_html=True)

    ids = sorted(df["CustomerID"].astype(int).tolist())
    cliente_sel = st.selectbox("ID do Cliente", options=ids, index=0,
                               key=selectbox_key, label_visibility="collapsed")

    reg = df[df["CustomerID"] == float(cliente_sel)].iloc[0]

    r_roi_val  = reg["ROI"] * 100
    r_roi_fmt  = f"{r_roi_val:+.2f}%"
    r_roi_cls  = "neg" if r_roi_val < 0 else "pos"
    r_luc_val  = reg["Lucro_Cliente"]
    r_luc_fmt  = f"R$ {r_luc_val:+,.2f}".replace(",", ".")
    r_luc_cls  = "neg" if r_luc_val < 0 else "pos"
    r_mon_fmt  = f"R$ {reg['Monetary']:,.2f}".replace(",", ".")
    r_cac_fmt  = f"R$ {reg['CAC']:,.2f}".replace(",", ".")
    r_dias_fmt = f"{int(reg['Dias_Sem_Comprar'])} dias"

    st.markdown(f"""
    <div class="exec-record-wrap">
      <div class="exec-record-title">
        👤 Registro do Cliente &nbsp;
        <span class="exec-badge">ID {int(reg['CustomerID'])}</span>
        &nbsp;
        <span class="exec-badge" style="background:#fff3e0;color:#b45309;border-color:#f6c26b;">{icone} {grupo_key}</span>
      </div>
      <div class="exec-grid">
        <div class="exec-cell"><div class="exec-cell-label">CustomerID</div><div class="exec-cell-value">{int(reg['CustomerID'])}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">CAC (R$)</div><div class="exec-cell-value">{r_cac_fmt}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Total Gasto (Monetary)</div><div class="exec-cell-value">{r_mon_fmt}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">ROI (%)</div><div class="exec-cell-value {r_roi_cls}">{r_roi_fmt}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Lucro Cliente (R$)</div><div class="exec-cell-value {r_luc_cls}">{r_luc_fmt}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Última Compra</div><div class="exec-cell-value">{reg['Ultima_Compra']}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Dias Sem Comprar</div><div class="exec-cell-value">{r_dias_fmt}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Recência</div><div class="exec-cell-value">{int(reg['Recency'])}d</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Frequência</div><div class="exec-cell-value">{int(reg['Frequency'])}x</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Seg. Recência</div><div class="exec-cell-value">{reg['Recency_segment']}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Seg. Frequência</div><div class="exec-cell-value">{reg['Frequency_segment']}</div></div>
        <div class="exec-cell"><div class="exec-cell-label">Seg. Monetário</div><div class="exec-cell-value">{reg['Monetary_segment']}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  3 TABELAS EXECUTIVAS SEPARADAS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Tabela Executiva de Clientes</div>', unsafe_allow_html=True)

render_tabela_executiva("Ocasional", "CLIENTES OCASIONAIS", "🕐", "sel_ocasional")
render_tabela_executiva("Regular",   "CLIENTES REGULARES",  "🔵", "sel_regular")
render_tabela_executiva("Engajado",  "CLIENTES ENGAJADOS",  "⭐", "sel_engajado")