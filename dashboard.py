import streamlit as st
import json
import os
import glob
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="🧪",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d0d0d;
        color: #e8e8e8;
    }
    .stApp { background-color: #0d0d0d; }

    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #00ff88;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }
    .answer-box {
        background: #111;
        border-left: 3px solid #00ff88;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #ccc;
        margin: 6px 0;
    }
    .answer-box.expected {
        border-left-color: #888;
        color: #aaa;
    }
    .score-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


def score_color(score):
    if score is None:
        return "#555"
    if score >= 0.8:
        return "#00ff88"
    elif score >= 0.5:
        return "#ffcc00"
    else:
        return "#ff4444"


def load_results():
    files = sorted(glob.glob("results/eval_*.json"), reverse=True)
    return files


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧪 LLM Evaluation Framework")
st.markdown("##### Accuracy · Hallucination · Relevance · Coherence")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Result Files")
    result_files = load_results()

    if not result_files:
        st.warning("No results found. Run `eval_runner.py` first.")
        st.stop()

    selected_file = st.selectbox("Select evaluation run", result_files)

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    with open(selected_file) as f:
        results = json.load(f)

    categories = ["All"] + sorted(set(r["category"] for r in results))
    selected_category = st.selectbox("Category", categories)

    min_score = st.slider("Min Overall Score", 0.0, 1.0, 0.0, 0.05)

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = results
if selected_category != "All":
    filtered = [r for r in filtered if r["category"] == selected_category]
filtered = [r for r in filtered if r["overall_score"] >= min_score]

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown("### 📊 Summary")

col1, col2, col3, col4, col5 = st.columns(5)

def avg(key_path):
    vals = []
    for r in results:
        keys = key_path.split(".")
        v = r
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            vals.append(v)
    return round(sum(vals) / len(vals), 3) if vals else 0

metrics = [
    ("Overall", avg("overall_score")),
    ("Accuracy", avg("accuracy.accuracy_score")),
    ("Hallucination", avg("hallucination.hallucination_score")),
    ("Relevance", avg("relevance.relevance_score")),
    ("Coherence", avg("coherence.coherence_score")),
]

for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        color = score_color(value)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{value:.3f}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ── Charts ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📈 Score Distribution")
    df = pd.DataFrame([{
        "Question": f"Q{r['id']}",
        "Accuracy": r["accuracy"]["accuracy_score"],
        "Hallucination": r["hallucination"]["hallucination_score"] or 0,
        "Relevance": r["relevance"]["relevance_score"] or 0,
        "Coherence": r["coherence"]["coherence_score"] or 0,
        "Overall": r["overall_score"]
    } for r in filtered])

    fig = go.Figure()
    colors = {"Accuracy": "#00ff88", "Hallucination": "#00ccff", "Relevance": "#ffcc00", "Coherence": "#ff88cc", "Overall": "#ffffff"}
    for metric, color in colors.items():
        fig.add_trace(go.Bar(name=metric, x=df["Question"], y=df[metric], marker_color=color, opacity=0.85))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="#111",
        paper_bgcolor="#1a1a1a",
        font=dict(family="IBM Plex Mono", color="#ccc", size=11),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### 🕸 Radar Chart (Averages)")
    categories_radar = ["Accuracy", "Hallucination", "Relevance", "Coherence"]
    values = [
        avg("accuracy.accuracy_score"),
        avg("hallucination.hallucination_score"),
        avg("relevance.relevance_score"),
        avg("coherence.coherence_score")
    ]

    fig2 = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories_radar + [categories_radar[0]],
        fill='toself',
        fillcolor='rgba(0,255,136,0.15)',
        line=dict(color='#00ff88', width=2),
        marker=dict(color='#00ff88', size=6)
    ))
    fig2.update_layout(
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(visible=True, range=[0, 1], color="#555", gridcolor="#222"),
            angularaxis=dict(color="#aaa", gridcolor="#222")
        ),
        plot_bgcolor="#111",
        paper_bgcolor="#1a1a1a",
        font=dict(family="IBM Plex Mono", color="#ccc", size=11),
        margin=dict(l=30, r=30, t=30, b=30),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Per-question results ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🔬 Individual Results ({len(filtered)} cases)")

for r in filtered:
    overall_color = score_color(r["overall_score"])
    with st.expander(f"Q{r['id']} · {r['category'].upper()} · Overall: {r['overall_score']}", expanded=False):
        st.markdown(f"**Question:** {r['question']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Expected Answer**")
            st.markdown(f'<div class="answer-box expected">{r["expected_answer"]}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("**LLM Answer**")
            st.markdown(f'<div class="answer-box">{r["actual_answer"]}</div>', unsafe_allow_html=True)

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)

        scores_data = [
            (m1, "Accuracy", r["accuracy"]["accuracy_score"], r["accuracy"].get("exact_match")),
            (m2, "Hallucination", r["hallucination"]["hallucination_score"], r["hallucination"].get("grounded")),
            (m3, "Relevance", r["relevance"]["relevance_score"], r["relevance"].get("reason")),
            (m4, "Coherence", r["coherence"]["coherence_score"], r["coherence"].get("reason")),
        ]

        for col, label, score, extra in scores_data:
            with col:
                color = score_color(score)
                val = f"{score:.3f}" if score is not None else "N/A"
                st.markdown(f"""
                <div style="text-align:center; padding:10px; background:#111; border-radius:6px; border:1px solid #222;">
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:{color}; font-weight:600">{val}</div>
                    <div style="font-size:0.7rem; color:#777; text-transform:uppercase; letter-spacing:0.08em">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        if r["hallucination"].get("reason"):
            st.caption(f"🔍 Hallucination: {r['hallucination']['reason']}")
        if r["relevance"].get("reason"):
            st.caption(f"🎯 Relevance: {r['relevance']['reason']}")
        if r["coherence"].get("reason"):
            st.caption(f"✍️ Coherence: {r['coherence']['reason']}")

# ── Raw data ──────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📄 Raw JSON Results"):
    st.json(filtered)