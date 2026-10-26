"""
Streamlit UI for the data analyst: pick or upload a CSV, ask questions, see the answer,
charts, verification and the code that produced it.
Run from the repo root:  streamlit run projects/data_analyst/ui.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `projects.` and guardrails

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from projects.data_analyst.graph import MAX_EXECUTIONS, SAMPLE_CSV, AnalystSession  # noqa: E402
from projects.research_report.graph import estimate_cost  # noqa: E402

st.set_page_config(page_title="Data Analyst", page_icon="📊", layout="wide")

EXAMPLES = [
    "Which region had the highest total revenue, and what share of the total was it?",
    "Plot monthly revenue by category. Which category is the most seasonal?",
    "Compare the average discount between Online and Store orders.",
    "Which product sold the most units in Q4?",
    "How many orders were over $1,000, and what share of revenue did they bring in?",
]


def describe(node: str, update: dict) -> str:
    if node == "understand" and update.get("spec"):
        s = update["spec"]
        return f"🧭 Plan: {s['intent']} · {', '.join(s['metrics']) or '-'} by {', '.join(s['dimensions']) or '-'}"
    if node == "executor" and update.get("executions"):
        e = update["executions"][-1]
        return f"🐍 Ran code #{len(update['executions'])} → {'ok' if e['ok'] else 'error, retrying'}"
    if node == "verify":
        v = update["verification"]
        return (f"🔍 Verified: {'all numbers grounded' if not v['ungrounded'] else 'ungrounded ' + ', '.join(v['ungrounded'])}"
                f" · method {'approved' if v['critic_approved'] else 'questioned'}")
    if node == "input_guard" and update.get("status") == "blocked":
        return "⛔ Question blocked by guardrails"
    return ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Data Analyst")
    upload = st.file_uploader("Upload a CSV (optional)", type="csv")
    source = upload.name if upload else SAMPLE_CSV.name
    if st.session_state.get("analyst_source") != source:
        path = SAMPLE_CSV
        if upload:
            path = Path(tempfile.mkdtemp()) / "upload.csv"
            path.write_bytes(upload.getvalue())
        st.session_state.update(analyst_source=source, analyst_session=AnalystSession(path), analyst_history=[])
    st.caption(f"Dataset: **{source}**")
    st.caption(f"Each question: up to {MAX_EXECUTIONS} sandboxed code runs, 1 revision after verification.")
    st.subheader("Try")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.analyst_pending = q

session: AnalystSession = st.session_state.analyst_session
with st.expander("Data preview", expanded=not st.session_state.analyst_history):
    st.dataframe(pd.read_csv(session.workdir / "data.csv", nrows=200), use_container_width=True, height=240)

# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question about the data...") or st.session_state.pop("analyst_pending", None)
if question:
    with st.status(f"Analyzing: {question}", expanded=True) as status:
        try:
            for node, update in session.stream(question):
                if line := describe(node, update):
                    st.write(line)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # network / API errors
            st.error(f"{type(exc).__name__}: {exc}")
            status.update(label="Failed", state="error")
    st.session_state.analyst_history.insert(0, (question, session.result))

# ---------------------------------------------------------------------------
# Answers (newest first)
# ---------------------------------------------------------------------------

for i, (q, r) in enumerate(st.session_state.analyst_history):
    st.markdown(f"### {q}")
    for note in r["guard_notes"]:
        st.caption(f"🛡️ {note}")
    if r["status"] == "blocked":
        st.error("Blocked by guardrails.")
        continue
    st.markdown(r["answer"])
    for chart in r["charts"]:
        st.image(chart)
    v = r["verification"]
    if r["status"] == "answered_with_concerns":
        st.warning("⚠️ The reviewer still had concerns after a revision; see the notes at the top of the answer.")
    elif r["status"] == "answered":
        if v.get("ungrounded"):
            st.warning(f"⚠️ Unverified numbers (not found in any code output): {', '.join(v['ungrounded'])}")
        else:
            st.success(f"✓ Every number traced to code output · method "
                       f"{'approved' if v.get('critic_approved') else 'not approved by reviewer'}"
                       + (f" · revised {r['revisions']}×" if r["revisions"] else ""))
    with st.expander(f"Show work ({len(r['executions'])} code run(s))"):
        for n, e in enumerate(r["executions"], 1):
            st.caption(f"Run {n} — {'ok' if e['ok'] else 'error'}")
            st.code(e["code"], language="python")
            st.code(e["output"], language="text")
    if i == 0:
        tokens = sum(u["input_tokens"] + u["output_tokens"] for u in r["usage"].values())
        st.caption(f"Session tokens: {tokens:,} (≈ ${estimate_cost(r['usage']):.4f})")
    st.divider()
