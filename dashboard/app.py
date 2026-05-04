"""
dashboard/app.py — HaloCheck Streamlit Dashboard

Reads:
  data/corpus_final.json   — annotated corpus with detections
  evaluation/results.json  — evaluation metrics (new schema)

Schema expected from evaluation/results.json:
  overall, per_type, per_severity, severity_accuracy, ood,
  hhem_baseline, stress_test, deduplication_check, corpus_stats, targets

Run: streamlit run dashboard/app.py
"""

import json
from pathlib import Path
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HaloCheck",
    page_icon="🔬",
    layout="wide",
)

CORPUS_PATH  = Path("data/corpus_final.json")
RESULTS_PATH = Path("evaluation/results.json")

SEVERITY_COLORS = {
    "Critical": "#FF4444",
    "Moderate": "#FF8C00",
    "Minor":    "#FFD700",
}

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_corpus():
    if not CORPUS_PATH.exists():
        return []
    with open(CORPUS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def highlight_detections(text, detections):
    """
    Wrap detected spans in colored HTML.
    All dict.get() calls are extracted into variables before the f-string
    to avoid the backslash-inside-braces SyntaxError.
    """
    result      = text
    sorted_dets = sorted(detections, key=lambda d: len(d.get("flagged_text", "")), reverse=True)
    for det in sorted_dets:
        flagged  = det.get("flagged_text", "")
        severity = det.get("severity", "Minor")
        color    = SEVERITY_COLORS.get(severity, "#FFD700")
        det_type = det.get("type", "")
        det_conf = det.get("confidence", 0)
        if flagged and flagged in result:
            span   = (
                f'<span style="background-color:{color}22;border:1px solid {color};'
                f'border-radius:3px;padding:1px 3px;font-weight:500;"'
                f' title="{det_type} | {severity} | conf={det_conf:.2f}">'
                f'{flagged}</span>'
            )
            result = result.replace(flagged, span, 1)
    return result


def metric_card(label, value, delta=None, good=True):
    color      = "#00C48C" if good else "#FF4444"
    delta_html = f'<div style="font-size:0.75rem;color:{color}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div style="background:#1E1E2E;border-radius:8px;padding:16px;text-align:center;'
        f'border:1px solid #2D2D3F;">'
        f'<div style="font-size:0.8rem;color:#888;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:1.8rem;font-weight:700;color:#FFF">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 HaloCheck")
    st.markdown("**Severity-Aware Hallucination Detection**")
    st.markdown("CS595 · IIT Chicago · Spring 2026")
    st.divider()

    page = st.radio(
        "View",
        ["📊 Metrics Dashboard", "🔍 Note Inspector", "📋 Corpus Browser"],
        label_visibility="collapsed",
    )


# ── Page: Metrics Dashboard ───────────────────────────────────────────────────

if page == "📊 Metrics Dashboard":
    st.title("📊 Evaluation Results")

    results = load_results()

    if not results:
        st.error(f"`{RESULTS_PATH}` not found. Run `python evaluation/evaluate.py` first.")
        st.stop()

    overall = results.get("overall", {})
    hhem    = results.get("hhem_baseline", {})
    ood     = results.get("ood", {})
    stress  = results.get("stress_test", {})
    targets = results.get("targets", {})
    dedup   = results.get("deduplication_check", {})
    stats   = results.get("corpus_stats", {})
    sev_acc = results.get("severity_accuracy", 0)

    # ── Corpus stats strip ────────────────────────────────────────────────────
    if stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total notes", stats.get("total",    "—"))
        c2.metric("Test set",    stats.get("test",     "—"))
        c3.metric("Stress set",  stats.get("stress",   "—"))
        c4.metric("Injected",    stats.get("injected", "—"))
        c5.metric("Pure OOD",    stats.get("pure_ood", "—"))
        st.divider()

    # ── HaloCheck vs Baseline ─────────────────────────────────────────────────
    st.subheader("HaloCheck vs facebook/bart-large-mnli Baseline")
    col1, col2, col3, col4 = st.columns(4)

    f1_val    = overall.get("f1", 0)
    p_val     = overall.get("precision", 0)
    r_val     = overall.get("recall", 0)
    hhem_f1   = hhem.get("f1", 0)
    target_f1 = targets.get("overall_f1_target", 0.70)
    target_p  = targets.get("precision_target",  0.72)
    target_r  = targets.get("recall_target",     0.68)

    with col1:
        tick = "✓" if f1_val >= target_f1 else "✗"
        metric_card("F1", f"{f1_val:.3f}", f"target >= {target_f1} {tick}", good=f1_val >= target_f1)
    with col2:
        tick = "✓" if p_val >= target_p else "✗"
        metric_card("Precision", f"{p_val:.3f}", f"target >= {target_p} {tick}", good=p_val >= target_p)
    with col3:
        tick = "✓" if r_val >= target_r else "✗"
        metric_card("Recall", f"{r_val:.3f}", f"target >= {target_r} {tick}", good=r_val >= target_r)
    with col4:
        if f1_val > hhem_f1:
            delta_str = f"HaloCheck +{f1_val - hhem_f1:.3f}"
        else:
            delta_str = "below baseline"
        metric_card("BART-NLI F1", f"{hhem_f1:.3f}", delta_str, good=f1_val > hhem_f1)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### Side-by-Side Comparison")
    import pandas as pd
    cmp_data = {
        "Metric":   ["Precision", "Recall", "F1", "TP", "FP", "FN"],
        "HaloCheck": [
            overall.get("precision", "—"), overall.get("recall", "—"),
            overall.get("f1",        "—"), overall.get("tp",     "—"),
            overall.get("fp",        "—"), overall.get("fn",     "—"),
        ],
        "BART-NLI": [
            hhem.get("precision", "—"), hhem.get("recall", "—"),
            hhem.get("f1",        "—"), hhem.get("tp",     "—"),
            hhem.get("fp",        "—"), hhem.get("fn",     "—"),
        ],
    }
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    # ── Per-type F1 ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Per-Type F1 (entity + type aligned)")
    per_type = results.get("per_type", {})
    if per_type:
        rows = []
        for t, m in sorted(per_type.items()):
            rows.append({
                "Injection Type": t,
                "F1":        m.get("f1",        0),
                "Precision": m.get("precision", 0),
                "Recall":    m.get("recall",    0),
                "TP": m.get("tp", 0),
                "FP": m.get("fp", 0),
                "FN": m.get("fn", 0),
            })
        df = pd.DataFrame(rows).sort_values("F1", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No per-type data. Run evaluate.py.")

    # ── OOD + Stress + Severity ───────────────────────────────────────────────
    st.divider()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("OOD (pure dosage_unit_swap)")
        if "note" in ood:
            st.warning(ood["note"])
        else:
            target_ood = targets.get("ood_f1_target", 0.45)
            ood_f1     = ood.get("f1", 0)
            tick       = "✓" if ood_f1 >= target_ood else "✗"
            st.metric("OOD F1",         f"{ood_f1:.3f}", f"target >= {target_ood} {tick}")
            st.metric("Pure OOD notes",  ood.get("pure_ood_notes", 0))
            st.metric("Precision",       f"{ood.get('precision', 0):.3f}")
            st.metric("Recall",          f"{ood.get('recall',    0):.3f}")

    with col_b:
        st.subheader("Domain Shift (Stress Test)")
        hc_stress = stress.get("halocheck", {})
        hh_stress = stress.get("hhem",      {})
        st.metric("HaloCheck stress F1", f"{hc_stress.get('f1', 0):.3f}")
        st.metric("BART-NLI stress F1",  f"{hh_stress.get('f1', 0):.3f}")

    with col_c:
        st.subheader("Severity")
        target_sa = targets.get("severity_acc_target", 0.75)
        tick      = "✓" if sev_acc >= target_sa else "✗"
        st.metric("Severity Accuracy", f"{sev_acc:.3f}", f"target >= {target_sa} {tick}")
        per_sev = results.get("per_severity", {})
        for sev in ["Critical", "Moderate", "Minor"]:
            m = per_sev.get(sev, {})
            if m:
                sev_color = SEVERITY_COLORS.get(sev, "#FFF")
                sev_f1    = m.get("f1", 0)
                st.markdown(
                    f'<span style="color:{sev_color}">**{sev}**</span> F1={sev_f1:.3f}',
                    unsafe_allow_html=True,
                )

    # ── Deduplication sanity ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Deduplication Sanity Check")
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("Total detections",   dedup.get("total_detections",   "—"))
    dcol2.metric("Multi-layer merged", dedup.get("multi_layer_merged", "—"))
    dcol3.metric("Merge rate",         f"{dedup.get('merge_rate', 0):.4f}")
    if dedup.get("merge_rate", 0) == 0.0 and dedup.get("total_detections", 0) > 0:
        st.warning("Merge rate is 0 — entity_id formats may still be misaligned between L1 and L2.")
    combos = dedup.get("layer_combinations", {})
    if combos:
        st.dataframe(
            pd.DataFrame(list(combos.items()), columns=["Layers", "Count"]),
            use_container_width=True,
            hide_index=True,
        )


# ── Page: Note Inspector ──────────────────────────────────────────────────────

elif page == "🔍 Note Inspector":
    st.title("🔍 Note Inspector")
    corpus = load_corpus()

    if not corpus:
        st.error(f"`{CORPUS_PATH}` not found. Run the pipeline first.")
        st.stop()

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        sources  = sorted({n.get("source", "?") for n in corpus})
        src_filt = st.multiselect("Source", sources, default=sources)
    with fcol2:
        sev_filt = st.multiselect(
            "Has severity",
            ["Critical", "Moderate", "Minor", "Clean"],
            default=["Critical", "Moderate", "Minor", "Clean"],
        )
    with fcol3:
        inj_filt = st.selectbox("Injection status", ["All", "Injected only", "Clean only"])

    filtered = corpus
    if src_filt:
        filtered = [n for n in filtered if n.get("source") in src_filt]
    if inj_filt == "Injected only":
        filtered = [n for n in filtered if n.get("is_injected")]
    elif inj_filt == "Clean only":
        filtered = [n for n in filtered if not n.get("is_injected")]
    if "Clean" not in sev_filt:
        filtered = [n for n in filtered if n.get("detections")]
    sev_set = {s for s in sev_filt if s != "Clean"}
    if sev_set:
        filtered = [
            n for n in filtered
            if any(d.get("severity") in sev_set for d in n.get("detections", []))
            or (not n.get("detections") and "Clean" in sev_filt)
        ]

    st.markdown(f"**{len(filtered)} notes match filters**")
    if not filtered:
        st.info("No notes match the current filters.")
        st.stop()

    note_ids    = [n["note_id"] for n in filtered]
    selected_id = st.selectbox("Select note", note_ids)
    note        = next(n for n in filtered if n["note_id"] == selected_id)

    # Header strip
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Source",     note.get("source", "?"))
    h2.metric("Injected",   "Yes" if note.get("is_injected") else "No")
    h3.metric("Pure OOD",   "Yes" if note.get("is_pure_ood") else "No")
    h4.metric("Detections", len(note.get("detections", [])))

    # Source | Summary side-by-side
    left, right = st.columns(2)

    source_html = note.get("source_text", "").replace("\n", "<br>")
    with left:
        st.markdown("### Source Note")
        st.markdown(
            f'<div style="background:#0D0D1A;border:1px solid #2D2D3F;border-radius:8px;'
            f'padding:16px;font-size:0.85rem;line-height:1.6;max-height:500px;overflow-y:auto">'
            f'{source_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### LLM Summary (test_summary)")
        highlighted      = highlight_detections(note.get("test_summary", ""), note.get("detections", []))
        highlighted_html = highlighted.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#0D0D1A;border:1px solid #2D2D3F;border-radius:8px;'
            f'padding:16px;font-size:0.85rem;line-height:1.8;max-height:500px;overflow-y:auto">'
            f'{highlighted_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Severity legend
    legend_parts = []
    for sev, col in SEVERITY_COLORS.items():
        legend_parts.append(
            f'<span style="background:{col}22;border:1px solid {col};'
            f'border-radius:3px;padding:2px 8px;font-size:0.8rem">{sev}</span>'
        )
    st.markdown(" &nbsp; ".join(legend_parts), unsafe_allow_html=True)

    # Detections table
    if note.get("detections"):
        st.markdown("#### Detections")
        import pandas as pd
        det_rows = []
        for d in note["detections"]:
            det_rows.append({
                "Severity":    d.get("severity", "?"),
                "Type":        d.get("type", "?"),
                "Flagged":     d.get("flagged_text", "")[:80],
                "Detected by": ", ".join(d.get("detected_by", [])),
                "Confidence":  f"{d.get('confidence', 0):.3f}",
                "entity_id":   d.get("entity_id", ""),
            })
        st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

    # Ground truth injections
    if note.get("injections"):
        st.markdown("#### Ground Truth Injections")
        for inj in note["injections"]:
            sev      = inj.get("severity", "?")
            color    = SEVERITY_COLORS.get(sev, "#888")
            inj_type = inj.get("type", "?")
            inj_eid  = inj.get("entity_id", "?")
            inj_orig = inj.get("original_text", "?")[:80]
            inj_to   = inj.get("injected_text", "?")[:80]
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:8px 12px;'
                f'background:#0D0D1A;border-radius:4px;margin:4px 0;font-size:0.85rem">'
                f'<b>{inj_type}</b> · {sev} · '
                f'entity_id: <code>{inj_eid}</code><br>'
                f'FROM: <code>{inj_orig}</code><br>'
                f'TO: <code>{inj_to}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # HHEM score
    hhem_score = note.get("hhem_score")
    if hhem_score is not None:
        flagged   = hhem_score < 0.5
        delta_str = "FLAGGED" if flagged else "CLEAN"
        delta_col = "inverse" if flagged else "normal"
        st.metric("BART-NLI score", f"{hhem_score:.4f}", delta=delta_str, delta_color=delta_col)

    # JSON export
    st.download_button(
        "⬇ Export this note as JSON",
        data=json.dumps(note, indent=2),
        file_name=f"{note['note_id']}_halocheck.json",
        mime="application/json",
    )


# ── Page: Corpus Browser ──────────────────────────────────────────────────────

elif page == "📋 Corpus Browser":
    st.title("📋 Corpus Browser")
    corpus = load_corpus()

    if not corpus:
        st.error(f"`{CORPUS_PATH}` not found.")
        st.stop()

    import pandas as pd

    rows = []
    for note in corpus:
        crits     = sum(1 for d in note.get("detections", []) if d.get("severity") == "Critical")
        mods      = sum(1 for d in note.get("detections", []) if d.get("severity") == "Moderate")
        inj_types = ", ".join({inj["type"] for inj in note.get("injections", [])})
        rows.append({
            "note_id":   note["note_id"],
            "source":    note.get("source", "?"),
            "split":     note.get("split",  "?"),
            "injected":  note.get("is_injected", False),
            "pure_ood":  note.get("is_pure_ood",  False),
            "inj_types": inj_types or "—",
            "critical":  crits,
            "moderate":  mods,
            "total_det": len(note.get("detections", [])),
            "hhem":      f"{note.get('hhem_score', 0):.3f}" if note.get("hhem_score") is not None else "—",
        })

    df = pd.DataFrame(rows)

    fc1, fc2, fc3 = st.columns(3)
    src_opts = sorted(df["source"].unique())
    sel_src  = fc1.multiselect("Source", src_opts, default=src_opts)
    spl_opts = sorted(df["split"].unique())
    sel_spl  = fc2.multiselect("Split",  spl_opts, default=spl_opts)
    inj_only = fc3.checkbox("Injected only", value=False)

    mask = df["source"].isin(sel_src) & df["split"].isin(sel_spl)
    if inj_only:
        mask &= df["injected"]
    df_filt = df[mask]

    st.markdown(f"**{len(df_filt)} notes**")
    st.dataframe(df_filt, use_container_width=True, hide_index=True, height=500)

    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Notes shown", len(df_filt))
    s2.metric("Injected",    int(df_filt["injected"].sum()))
    s3.metric("Total crits", int(df_filt["critical"].sum()))
    s4.metric("Total mods",  int(df_filt["moderate"].sum()))

    filtered_ids = set(df_filt["note_id"])
    st.download_button(
        "⬇ Export filtered corpus as JSON",
        data=json.dumps(
            [n for n in corpus if n["note_id"] in filtered_ids],
            indent=2,
        ),
        file_name="halocheck_corpus_filtered.json",
        mime="application/json",
    )
