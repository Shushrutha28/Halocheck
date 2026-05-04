"""
dashboard/app.py — HaloCheck Interactive Dashboard (Redesigned)
Run: streamlit run dashboard/app.py
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HaloCheck · Clinical Hallucination Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORPUS_PATH  = Path("data/corpus_final.json")
RESULTS_PATH = Path("evaluation/results.json")

SEVERITY_HEX = {
    "Critical": "#f85149",
    "Moderate": "#e3b341",
    "Minor":    "#3fb950",
}
SEVERITY_ALPHA = {
    "Critical": "rgba(248,81,73,0.15)",
    "Moderate": "rgba(227,179,65,0.15)",
    "Minor":    "rgba(63,185,80,0.15)",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:     #0d1117; --surf: #161b22; --surf2: #1c2330; --surf3: #21262d;
    --border: #30363d; --text: #e6edf3; --muted:  #8b949e;
    --accent: #58a6ff; --green: #3fb950; --yellow: #e3b341;
    --red: #f85149;    --purple: #bc8cff;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}
html, body, [class*="css"] { font-family: var(--sans) !important; background: var(--bg) !important; color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 1500px; }
section[data-testid="stSidebar"] { background: var(--surf) !important; border-right: 1px solid var(--border); }

/* SIDEBAR */
.sb-logo { font-family: var(--mono); font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.sb-sub  { font-size: 0.72rem; color: var(--muted); line-height: 1.5; margin-bottom: 14px; }

/* PAGE TITLE */
.page-title { font-family: var(--mono); font-size: 1.25rem; font-weight: 600; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 18px; display: flex; align-items: center; gap: 10px; }
.chip { font-size: 0.63rem; background: var(--surf2); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; font-weight: 400; }

/* SECTION LABEL */
.sec { font-size: 0.67rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 5px; margin: 16px 0 10px; }

/* KPI CARD */
.kpi-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.kpi { flex: 1; min-width: 120px; background: var(--surf); border: 1px solid var(--border); border-radius: 8px; padding: 13px 15px; position: relative; overflow: hidden; }
.kpi::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: var(--kc, var(--accent)); }
.kpi:hover { border-color: var(--accent); }
.kpi .lbl { font-size: 0.67rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 5px; }
.kpi .val { font-family: var(--mono); font-size: 1.85rem; font-weight: 700; line-height: 1; color: var(--kc, var(--text)); }
.kpi .sub { font-size: 0.69rem; color: var(--muted); margin-top: 4px; }
.k-blue   { --kc: var(--accent); } .k-green  { --kc: var(--green); }
.k-yellow { --kc: var(--yellow); } .k-red    { --kc: var(--red); }
.k-purple { --kc: var(--purple); } .k-muted  { --kc: var(--muted); }

/* DETECTION CARD */
.dc { border-left: 3px solid; border-radius: 0 6px 6px 0; padding: 10px 14px; margin-bottom: 8px; font-size: 0.83rem; }
.dc-Critical { border-color: var(--red);    background: rgba(248,81,73,0.08); }
.dc-Moderate { border-color: var(--yellow); background: rgba(227,179,65,0.08); }
.dc-Minor    { border-color: var(--green);  background: rgba(63,185,80,0.08); }
.dbg { display:inline-block; font-family:var(--mono); font-size:0.64rem; font-weight:600; padding:1px 7px; border-radius:3px; text-transform:uppercase; letter-spacing:0.4px; margin-right:7px; }
.dbg-Critical { background:rgba(248,81,73,0.2);  color:var(--red); }
.dbg-Moderate { background:rgba(227,179,65,0.2); color:var(--yellow); }
.dbg-Minor    { background:rgba(63,185,80,0.2);  color:var(--green); }
.dt   { color:var(--muted); font-size:0.75rem; }
.dtxt { color:var(--text); font-style:italic; margin:5px 0 3px; }
.dmeta{ color:var(--muted); font-family:var(--mono); font-size:0.71rem; }
.cbar { background:var(--surf3); border-radius:3px; height:4px; margin-top:6px; }
.cfill{ height:100%; border-radius:3px; }

/* NOTE BOX */
.nbox { background:var(--surf2); border:1px solid var(--border); border-radius:8px; padding:15px; font-size:0.82rem; line-height:1.7; max-height:440px; overflow-y:auto; white-space:pre-wrap; }
.nbox::-webkit-scrollbar { width:4px; }
.nbox::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

/* INJECTION CARD */
.ic { border-left:3px solid; border-radius:0 6px 6px 0; background:var(--surf2); padding:10px 14px; margin:6px 0; font-size:0.81rem; }

/* PILLS */
.pill { display:inline-block; font-size:0.71rem; font-weight:500; padding:2px 10px; border-radius:20px; border:1px solid; }
.p-inj   { color:var(--red);    background:rgba(248,81,73,0.1);  border-color:rgba(248,81,73,0.3); }
.p-clean { color:var(--green);  background:rgba(63,185,80,0.1);  border-color:rgba(63,185,80,0.3); }
.p-ood   { color:var(--purple); background:rgba(188,140,255,0.1);border-color:rgba(188,140,255,0.3); }

/* COMPARISON BAR */
.cr { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.cn { font-size:0.77rem; color:var(--muted); min-width:110px; text-align:right; }
.cv { font-family:var(--mono); font-size:0.86rem; min-width:48px; }
.ct { flex:1; background:var(--surf3); border-radius:3px; height:7px; overflow:hidden; }
.cf { height:100%; border-radius:3px; }

/* TABLES */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden; }
[data-testid="metric-container"] { background:var(--surf) !important; border:1px solid var(--border) !important; border-radius:8px !important; padding:12px 14px !important; }
[data-testid="metric-container"] label { color:var(--muted) !important; font-size:0.69rem !important; text-transform:uppercase; letter-spacing:1px; }
[data-testid="stMetricValue"] { font-family:var(--mono) !important; font-size:1.45rem !important; color:var(--accent) !important; }
hr { border-color: var(--border) !important; }
.stSelectbox > div > div, .stMultiSelect > div > div { background:var(--surf) !important; border:1px solid var(--border) !important; border-radius:6px !important; }
.stSelectbox label, .stMultiSelect label { font-size:0.69rem !important; text-transform:uppercase !important; letter-spacing:1px !important; color:var(--muted) !important; }
.stDownloadButton button { background:var(--surf) !important; border:1px solid var(--accent) !important; color:var(--accent) !important; border-radius:6px !important; font-size:0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_corpus() -> List[Dict]:
    if not CORPUS_PATH.exists():
        return []
    with open(CORPUS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_results() -> Optional[Dict]:
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def highlight_detections(text: str, detections: List[Dict]) -> str:
    result = text
    for det in sorted(detections, key=lambda d: len(d.get("flagged_text", "")), reverse=True):
        flagged  = det.get("flagged_text", "")
        sev      = det.get("severity", "Minor")
        color    = SEVERITY_HEX.get(sev, "#888")
        bg       = SEVERITY_ALPHA.get(sev, "rgba(128,128,128,0.1)")
        tip      = f"{det.get('type','')} | {sev} | conf={det.get('confidence',0):.2f}"
        if flagged and flagged in result:
            span = (f'<mark style="background:{bg};border:1px solid {color};'
                    f'border-radius:3px;padding:0 3px;font-weight:500;color:{color}"'
                    f' title="{tip}">{flagged}</mark>')
            result = result.replace(flagged, span, 1)
    return result


def pl(base: dict, **kwargs) -> dict:
    out = dict(base)
    out.update(kwargs)
    return out


def gauge(value: float, title: str, target: float, color: str) -> go.Figure:
    tick = "✓" if value >= target else "✗"
    fig  = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 4),
        title={"text": f"{title}<br><span style='font-size:0.72rem;color:#8b949e'>target {target} {tick}</span>",
               "font": {"size": 13, "color": "#c9d1d9"}},
        number={"font": {"size": 26, "family": "IBM Plex Mono", "color": color}},
        gauge={
            "axis":  {"range": [0, 1], "tickwidth": 0, "tickcolor": "#30363d",
                      "tickvals": [0, 0.5, 1.0], "ticktext": ["0", ".5", "1"]},
            "bar":   {"color": color, "thickness": 0.22},
            "bgcolor": "#161b22", "borderwidth": 0,
            "steps": [{"range": [0, target], "color": "#21262d"},
                      {"range": [target, 1.0], "color": "#1c2330"}],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": target},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Mono", color="#c9d1d9"),
                      height=195, margin=dict(t=60, b=5, l=15, r=15))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">🧬 HaloCheck</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Severity-Aware Hallucination Detection<br>for LLM Clinical Summaries<br><span style="color:#58a6ff">CS595 · IIT · Spring 2026</span></div>', unsafe_allow_html=True)
    st.markdown('<hr style="margin:6px 0 14px">', unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["📊  Metrics Dashboard", "🔍  Note Inspector", "📋  Corpus Browser"],
                    label_visibility="collapsed")

    r_quick = load_results()
    if r_quick:
        ov = r_quick.get("overall", {})
        cs = r_quick.get("corpus_stats", {})
        st.markdown('<hr style="margin:14px 0 8px">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.64rem;text-transform:uppercase;letter-spacing:1.2px;color:#8b949e;margin-bottom:6px">Quick Stats</div>', unsafe_allow_html=True)
        _a, _b = st.columns(2)
        _a.metric("F1",  f"{ov.get('f1',0):.3f}")
        _b.metric("Rec", f"{ov.get('recall',0):.3f}")
        if cs:
            _c, _d = st.columns(2)
            _c.metric("Notes", cs.get("total","—"))
            _d.metric("Inj",   cs.get("injected","—"))


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — METRICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊  Metrics Dashboard":
    results = load_results()
    st.markdown('<div class="page-title">📊 Evaluation Dashboard <span class="chip">v2 corpus</span></div>', unsafe_allow_html=True)
    if not results:
        st.error("Run `python evaluation/evaluate.py` first.")
        st.stop()

    overall  = results.get("overall", {})
    hhem     = results.get("hhem_baseline", {})
    ood      = results.get("ood", {})
    stress   = results.get("stress_test", {})
    targets  = results.get("targets", {})
    dedup    = results.get("deduplication_check", {})
    stats    = results.get("corpus_stats", {})
    sev_acc  = results.get("severity_accuracy", 0)
    per_type = results.get("per_type", {})
    per_sev  = results.get("per_severity", {})

    f1_val  = overall.get("f1", 0)
    p_val   = overall.get("precision", 0)
    r_val   = overall.get("recall", 0)
    hhem_f1 = hhem.get("f1", 0)

    # ── Corpus strip ──────────────────────────────────────────────────────────
    if stats:
        st.markdown('<div class="sec">Corpus</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi k-muted"><div class="lbl">Total</div><div class="val">{stats.get("total","—")}</div></div>
          <div class="kpi k-muted"><div class="lbl">Test Set</div><div class="val">{stats.get("test","—")}</div></div>
          <div class="kpi k-muted"><div class="lbl">Stress Set</div><div class="val">{stats.get("stress","—")}</div></div>
          <div class="kpi k-yellow"><div class="lbl">Injected</div><div class="val">{stats.get("injected","—")}</div></div>
          <div class="kpi k-purple"><div class="lbl">Pure OOD</div><div class="val">{stats.get("pure_ood","—")}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauges row ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">HaloCheck Performance — Test Set</div>', unsafe_allow_html=True)
    _t_f1 = targets.get("overall_f1_target", 0.7)
    _t_p  = targets.get("precision_target",  0.72)
    _t_r  = targets.get("recall_target",     0.68)
    _t_oo = targets.get("ood_f1_target",     0.45)
    _t_sa = targets.get("severity_acc_target",0.75)

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.plotly_chart(gauge(f1_val,          "F1 Score",   _t_f1, "#58a6ff"), use_container_width=True)
    g2.plotly_chart(gauge(p_val,           "Precision",  _t_p,  "#e3b341"), use_container_width=True)
    g3.plotly_chart(gauge(r_val,           "Recall",     _t_r,  "#3fb950"), use_container_width=True)
    g4.plotly_chart(gauge(ood.get("f1",0), "OOD F1",    _t_oo, "#bc8cff"), use_container_width=True)
    g5.plotly_chart(gauge(sev_acc,         "Sev. Acc.", _t_sa, "#79c0ff"), use_container_width=True)

    # ── TP/FP/FN + HHEM chips ─────────────────────────────────────────────────
    tp, fp, fn = overall.get("tp",0), overall.get("fp",0), overall.get("fn",0)
    impr = round(f1_val - hhem_f1, 3)
    st.markdown(f"""
    <div style="display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap">
      <div class="kpi k-green"  style="flex:0 0 auto;min-width:100px;text-align:center"><div class="lbl">True Pos</div><div class="val">{tp}</div></div>
      <div class="kpi k-red"    style="flex:0 0 auto;min-width:100px;text-align:center"><div class="lbl">False Pos</div><div class="val">{fp}</div></div>
      <div class="kpi k-yellow" style="flex:0 0 auto;min-width:100px;text-align:center"><div class="lbl">False Neg</div><div class="val">{fn}</div></div>
      <div class="kpi k-blue"   style="flex:0 0 auto;min-width:130px;text-align:center"><div class="lbl">vs BART-MNLI</div><div class="val">{hhem_f1:.3f}</div><div class="sub">+{impr:.3f} HC advantage</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-type bar chart ────────────────────────────────────────────────────
    st.markdown('<div class="sec">Per-Type F1</div>', unsafe_allow_html=True)
    types_with_sup = [(t, m) for t, m in per_type.items() if m.get("tp",0) + m.get("fn",0) > 0]
    if types_with_sup:
        types_sorted = sorted(types_with_sup, key=lambda x: -x[1]["f1"])
        t_labels = [t.replace("_"," ") for t, _ in types_sorted]
        hc_f1s   = [m["f1"]            for _, m in types_sorted]
        hh_f1s   = [results.get("per_type",{}).get(t,{}).get("f1",0) for t, _ in types_sorted]
        bar_colors = [SEVERITY_HEX["Minor"] if v >= 0.6 else
                      SEVERITY_HEX["Moderate"] if v >= 0.3 else
                      SEVERITY_HEX["Critical"] for v in hc_f1s]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="HaloCheck", x=t_labels, y=hc_f1s,
            marker=dict(color=bar_colors, opacity=0.85),
            text=[f"{v:.3f}" for v in hc_f1s],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9"),
        ))
        fig_bar.add_hline(y=_t_f1, line_dash="dash", line_color="#58a6ff",
                          annotation_text=f"target {_t_f1}",
                          annotation_font=dict(color="#58a6ff", size=10))
        fig_bar.update_layout(**pl(PLOTLY_LAYOUT,
            title=dict(text="F1 Score per Injection Type", font=dict(size=13), x=0),
            xaxis=dict(tickangle=-20, **PLOTLY_LAYOUT["xaxis"]),
            yaxis=dict(range=[0, 1.15], **PLOTLY_LAYOUT["yaxis"]),
            height=300, showlegend=False, bargap=0.32,
            margin=dict(t=40, b=10, l=10, r=10),
        ))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── HaloCheck vs HHEM grouped bar ─────────────────────────────────────────
    st.markdown('<div class="sec">HaloCheck vs BART-MNLI Baseline</div>', unsafe_allow_html=True)
    metrics  = ["Precision", "Recall", "F1"]
    hc_vals  = [p_val, r_val, f1_val]
    hh_vals  = [hhem.get("precision",0), hhem.get("recall",0), hhem_f1]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="HaloCheck", x=metrics, y=hc_vals,
                             marker_color="#58a6ff", opacity=0.9,
                             text=[f"{v:.3f}" for v in hc_vals], textposition="outside",
                             textfont=dict(family="IBM Plex Mono", size=12)))
    fig_cmp.add_trace(go.Bar(name="BART-MNLI", x=metrics, y=hh_vals,
                             marker_color="#8b949e", opacity=0.75,
                             text=[f"{v:.3f}" for v in hh_vals], textposition="outside",
                             textfont=dict(family="IBM Plex Mono", size=12)))
    fig_cmp.update_layout(**pl(PLOTLY_LAYOUT,
        barmode="group", bargap=0.28, bargroupgap=0.06,
        yaxis=dict(range=[0, 1.18], **PLOTLY_LAYOUT["yaxis"]),
        height=270,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=10, l=10, r=10),
    ))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Bottom row: Stress | OOD | Severity | Dedup ───────────────────────────
    st.markdown('<div class="sec">Generalisation · OOD · Severity · Deduplication</div>', unsafe_allow_html=True)
    bc1, bc2, bc3, bc4 = st.columns(4, gap="large")

    with bc1:
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin-bottom:8px">Stress Test (PubMed)</div>', unsafe_allow_html=True)
        hc_s = stress.get("halocheck", {})
        hh_s = stress.get("hhem", {})
        for label, vals, color in [("HaloCheck", hc_s, "#58a6ff"), ("BART-MNLI", hh_s, "#8b949e")]:
            f = vals.get("f1", 0)
            st.markdown(f'<div class="cr"><div class="cn">{label}</div><div class="ct"><div class="cf" style="width:{int(f*100)}%;background:{color}"></div></div><div class="cv" style="color:{color}">{f:.3f}</div></div>', unsafe_allow_html=True)

    with bc2:
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin-bottom:8px">OOD (dosage_unit_swap)</div>', unsafe_allow_html=True)
        for lbl, val, col in [("F1", ood.get("f1",0), "#bc8cff"), ("Prec", ood.get("precision",0), "#79c0ff"), ("Rec", ood.get("recall",0), "#3fb950")]:
            st.markdown(f'<div class="cr"><div class="cn">{lbl}</div><div class="ct"><div class="cf" style="width:{int(val*100)}%;background:{col}"></div></div><div class="cv" style="color:{col}">{val:.3f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.7rem;color:#8b949e">{ood.get("pure_ood_notes",0)} pure OOD notes</div>', unsafe_allow_html=True)

    with bc3:
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin-bottom:8px">Per Severity Tier</div>', unsafe_allow_html=True)
        for sev in ["Critical", "Moderate", "Minor"]:
            m   = per_sev.get(sev, {})
            f   = m.get("f1", 0)
            col = SEVERITY_HEX.get(sev, "#888")
            st.markdown(f'<div class="cr"><div class="cn" style="color:{col}">{sev}</div><div class="ct"><div class="cf" style="width:{int(f*100)}%;background:{col}"></div></div><div class="cv" style="color:{col}">{f:.3f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.7rem;color:#8b949e">Sev. Accuracy: {sev_acc:.3f}</div>', unsafe_allow_html=True)

    with bc4:
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin-bottom:8px">Deduplication</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid" style="flex-direction:column;gap:8px">
          <div class="kpi k-blue" style="text-align:center;padding:10px 12px"><div class="lbl">Total Det.</div><div class="val" style="font-size:1.4rem">{dedup.get("total_detections","—")}</div></div>
          <div class="kpi k-green" style="text-align:center;padding:10px 12px"><div class="lbl">Merged</div><div class="val" style="font-size:1.4rem">{dedup.get("multi_layer_merged","—")}</div></div>
        </div>
        <div style="font-size:0.7rem;color:#8b949e;margin-top:4px">Merge rate: {dedup.get("merge_rate",0):.4f}</div>
        """, unsafe_allow_html=True)

    # ── Full per-type table ────────────────────────────────────────────────────
    with st.expander("Full Per-Type Detail Table"):
        rows = []
        for t, m in sorted(per_type.items(), key=lambda x: -x[1].get("f1",0)):
            rows.append({"Type": t, "F1": round(m.get("f1",0),4),
                         "Precision": round(m.get("precision",0),4),
                         "Recall": round(m.get("recall",0),4),
                         "TP": m.get("tp",0), "FP": m.get("fp",0), "FN": m.get("fn",0)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Layer combo table ──────────────────────────────────────────────────────
    combos = dedup.get("layer_combinations", {})
    if combos:
        with st.expander("Layer Detection Breakdown"):
            st.dataframe(
                pd.DataFrame(list(combos.items()), columns=["Layers", "Count"]),
                use_container_width=True, hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — NOTE INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍  Note Inspector":
    corpus = load_corpus()
    st.markdown('<div class="page-title">🔍 Note Inspector <span class="chip">side-by-side diff</span></div>', unsafe_allow_html=True)
    if not corpus:
        st.error(f"`{CORPUS_PATH}` not found.")
        st.stop()

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4, gap="medium")
    sources  = sorted({n.get("source","?") for n in corpus})
    src_filt = fc1.multiselect("Source", sources, default=sources)
    inj_filt = fc2.selectbox("Status", ["All", "Injected only", "Clean only"])
    sev_filt = fc3.multiselect("Severity", ["Critical","Moderate","Minor","Clean"],
                               default=["Critical","Moderate","Minor","Clean"])
    ood_only = fc4.checkbox("Pure OOD only")

    filtered = corpus
    if src_filt:    filtered = [n for n in filtered if n.get("source") in src_filt]
    if inj_filt == "Injected only": filtered = [n for n in filtered if n.get("is_injected")]
    elif inj_filt == "Clean only":  filtered = [n for n in filtered if not n.get("is_injected")]
    if ood_only:    filtered = [n for n in filtered if n.get("is_pure_ood")]
    sev_set = {s for s in sev_filt if s != "Clean"}
    if sev_set:
        filtered = [
            n for n in filtered
            if any(d.get("severity") in sev_set for d in n.get("detections",[]))
            or (not n.get("detections") and "Clean" in sev_filt)
        ]

    st.markdown(f'<div style="font-size:0.74rem;color:#8b949e;margin-bottom:8px">{len(filtered)} notes matched</div>', unsafe_allow_html=True)
    if not filtered:
        st.info("No notes match current filters.")
        st.stop()

    selected_id = st.selectbox("Select note", [n["note_id"] for n in filtered])
    note        = next(n for n in filtered if n["note_id"] == selected_id)

    is_inj = note.get("is_injected", False)
    is_ood = note.get("is_pure_ood", False)
    n_dets = len(note.get("detections", []))

    pills = ('<span class="pill p-inj">⚠ Injected</span>' if is_inj else '<span class="pill p-clean">✓ Clean</span>')
    if is_ood: pills += ' <span class="pill p-ood">◈ Pure OOD</span>'

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap">'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.83rem;color:#8b949e;background:#1c2330;padding:2px 8px;border-radius:4px;border:1px solid #30363d">{note["note_id"]}</span>'
        f'<span style="font-size:0.77rem;color:#8b949e">{note.get("source","?")} · {note.get("split","?")} split</span>'
        f'{pills}'
        f'<span style="font-size:0.77rem;color:#58a6ff;margin-left:auto">{n_dets} detection(s)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Source | Summary side by side
    lc, rc = st.columns(2, gap="large")
    with lc:
        st.markdown('<div class="sec">Source Note (Ground Truth)</div>', unsafe_allow_html=True)
        src = (note.get("source_text") or note.get("text",""))[:4000]
        st.markdown(f'<div class="nbox">{src.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

    with rc:
        st.markdown('<div class="sec">LLM Test Summary — detections highlighted</div>', unsafe_allow_html=True)
        test_text   = note.get("test_summary","")
        highlighted = highlight_detections(test_text, note.get("detections",[]))
        st.markdown(f'<div class="nbox">{highlighted.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
        legend = " &nbsp; ".join([
            f'<span style="background:{SEVERITY_ALPHA[s]};border:1px solid {SEVERITY_HEX[s]};border-radius:3px;padding:1px 7px;font-size:0.73rem;color:{SEVERITY_HEX[s]}">{s}</span>'
            for s in ["Critical","Moderate","Minor"]
        ])
        st.markdown(legend, unsafe_allow_html=True)

    # Detection cards
    dets = note.get("detections", [])
    if dets:
        st.markdown('<div class="sec">Pipeline Detections</div>', unsafe_allow_html=True)
        for d in dets:
            sev   = d.get("severity","Minor")
            color = SEVERITY_HEX.get(sev,"#888")
            conf  = d.get("confidence", 0)
            st.markdown(f"""
            <div class="dc dc-{sev}">
              <span class="dbg dbg-{sev}">{sev}</span>
              <span class="dt">{d.get("type","")}</span>
              <div class="dtxt">"{d.get("flagged_text","")[:160]}"</div>
              <div class="dmeta">Layers: {" + ".join(d.get("detected_by",[]))} &nbsp;·&nbsp; Confidence: {conf:.3f}</div>
              <div class="cbar"><div class="cfill" style="width:{int(conf*100)}%;background:{color}"></div></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#8b949e;font-size:0.84rem;padding:8px 0">No detections by HaloCheck pipeline.</div>', unsafe_allow_html=True)

    # Ground truth injections
    injs = note.get("injections", [])
    if injs:
        st.markdown('<div class="sec">Ground Truth Injections</div>', unsafe_allow_html=True)
        for inj in injs:
            sev   = inj.get("severity","?")
            color = SEVERITY_HEX.get(sev, "#888")
            orig  = inj.get("original_text","?")[:120]
            injtx = inj.get("injected_text","?")[:120]
            st.markdown(f"""
            <div class="ic" style="border-color:{color}">
              <span class="dbg dbg-{sev}">{sev}</span>
              <strong style="color:{color}">{inj.get("type","")}</strong>
              <span style="font-size:0.74rem;color:#8b949e"> · entity_id: {inj.get("entity_id","?")}</span>
              <div style="margin-top:6px;font-size:0.8rem">
                FROM: <code style="background:#21262d;padding:1px 5px;border-radius:3px;color:#8b949e">{orig}</code><br>
                TO: &nbsp; <code style="background:#21262d;padding:1px 5px;border-radius:3px;color:{color}">{injtx}</code>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # HHEM score
    hhem_score = note.get("hhem_score")
    if hhem_score is not None:
        flagged  = hhem_score < 0.5
        flag_col = "#f85149" if flagged else "#3fb950"
        flag_txt = "🚨 FLAGGED" if flagged else "✓ CLEAN"
        st.markdown(f'<div style="font-size:0.79rem;color:#8b949e;margin:10px 0">BART-MNLI baseline: <span style="font-family:IBM Plex Mono,monospace;color:{flag_col}">{hhem_score:.4f} — {flag_txt}</span></div>', unsafe_allow_html=True)

    st.download_button("⬇  Export this note as JSON",
                       data=json.dumps(note, indent=2),
                       file_name=f"{note['note_id']}_halocheck.json",
                       mime="application/json")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — CORPUS BROWSER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋  Corpus Browser":
    corpus = load_corpus()
    st.markdown('<div class="page-title">📋 Corpus Browser <span class="chip">1008 notes</span></div>', unsafe_allow_html=True)
    if not corpus:
        st.error(f"`{CORPUS_PATH}` not found.")
        st.stop()

    rows = []
    for note in corpus:
        crits     = sum(1 for d in note.get("detections",[]) if d.get("severity")=="Critical")
        mods      = sum(1 for d in note.get("detections",[]) if d.get("severity")=="Moderate")
        inj_types = ", ".join(sorted({inj["type"] for inj in note.get("injections",[])}))
        rows.append({
            "note_id":   note["note_id"],
            "source":    note.get("source","?"),
            "split":     note.get("split","?"),
            "injected":  note.get("is_injected",False),
            "pure_ood":  note.get("is_pure_ood",False),
            "inj_types": inj_types or "—",
            "critical":  crits,
            "moderate":  mods,
            "total_det": len(note.get("detections",[])),
            "hhem":      round(note["hhem_score"],3) if note.get("hhem_score") is not None else None,
        })
    df = pd.DataFrame(rows)

    fc1, fc2, fc3, fc4 = st.columns(4, gap="medium")
    src_opts = sorted(df["source"].unique())
    spl_opts = sorted(df["split"].unique())
    sel_src  = fc1.multiselect("Source", src_opts, default=src_opts)
    sel_spl  = fc2.multiselect("Split",  spl_opts, default=spl_opts)
    inj_only = fc3.checkbox("Injected only")
    ood_only = fc4.checkbox("Pure OOD only")

    mask = df["source"].isin(sel_src) & df["split"].isin(sel_spl)
    if inj_only: mask &= df["injected"]
    if ood_only: mask &= df["pure_ood"]
    df_filt = df[mask]

    total_f = len(df_filt)
    inj_f   = int(df_filt["injected"].sum())
    crit_f  = int(df_filt["critical"].sum())
    mod_f   = int(df_filt["moderate"].sum())

    st.markdown(f"""
    <div class="kpi-grid" style="margin-bottom:12px">
      <div class="kpi k-blue"  ><div class="lbl">Shown</div><div class="val">{total_f}</div></div>
      <div class="kpi k-yellow"><div class="lbl">Injected</div><div class="val">{inj_f}</div></div>
      <div class="kpi k-red"   ><div class="lbl">Critical Det</div><div class="val">{crit_f}</div></div>
      <div class="kpi k-yellow"><div class="lbl">Moderate Det</div><div class="val">{mod_f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Source + injection type charts
    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        src_counts = df_filt["source"].value_counts()
        fig_src = go.Figure(go.Bar(
            x=src_counts.index.tolist(), y=src_counts.values.tolist(),
            marker_color="#58a6ff", opacity=0.85,
            text=src_counts.values.tolist(), textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9"),
        ))
        fig_src.update_layout(**pl(PLOTLY_LAYOUT,
            title=dict(text="Notes by Source", font=dict(size=13), x=0),
            height=230, showlegend=False, bargap=0.35,
            margin=dict(t=40, b=10, l=10, r=10),
        ))
        st.plotly_chart(fig_src, use_container_width=True)

    with ch2:
        inj_counts = df_filt["inj_types"].value_counts().head(8)
        fig_inj = go.Figure(go.Bar(
            y=inj_counts.index.tolist(), x=inj_counts.values.tolist(),
            orientation="h",
            marker_color="#e3b341", opacity=0.85,
            text=inj_counts.values.tolist(), textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9"),
        ))
        fig_inj.update_layout(**pl(PLOTLY_LAYOUT,
            title=dict(text="Injection Type Distribution", font=dict(size=13), x=0),
            height=230, showlegend=False,
            xaxis=dict(range=[0, max(inj_counts.values)*1.25], **PLOTLY_LAYOUT["xaxis"]),
            margin=dict(t=40, b=10, l=10, r=10),
        ))
        st.plotly_chart(fig_inj, use_container_width=True)

    # Main table
    st.markdown('<div class="sec">Notes Table</div>', unsafe_allow_html=True)
    st.dataframe(df_filt, use_container_width=True, hide_index=True, height=460)

    st.markdown('<div class="sec">Export</div>', unsafe_allow_html=True)
    filtered_ids = set(df_filt["note_id"])
    st.download_button(
        "⬇  Export filtered corpus as JSON",
        data=json.dumps([n for n in corpus if n["note_id"] in filtered_ids], indent=2),
        file_name="halocheck_corpus_filtered.json",
        mime="application/json",
    )
