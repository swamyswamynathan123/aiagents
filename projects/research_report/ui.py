"""
Streamlit UI for the research report generator: live progress, optional plan review, report + metrics.
Run from the repo root:  streamlit run projects/research_report/ui.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `projects.` and guardrails

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from projects.research_report.graph import MAX_QUESTIONS, ReportSession, estimate_cost  # noqa: E402

st.set_page_config(page_title="Research Reports", page_icon="📑", layout="wide")


def describe(node: str, update: dict) -> str:
    if node == "planner":
        return f"📝 Planned {len(update['plan'])} research questions"
    if node == "plan_review":
        return "▶️ Plan approved, researching in parallel"
    if node == "research":
        f = update["findings"][0]
        blocked = len(update.get("blocked_sources", []))
        return f"🔎 {f['question'][:80]} — {len(f['facts'])} facts" + (f" · 🛡️ {blocked} source(s) dropped" if blocked else "")
    if node == "writer":
        return f"✍️ Draft written ({len(update['draft'].split())} words, {len(update['references'])} sources)"
    if node == "reviewer":
        r = update["reviews"][-1]
        return f"🧐 Review {r['score']}/10 → {r['decision']}" + (f" · gaps: {', '.join(r['gaps'])}" if r["gaps"] else "")
    if node == "finalize":
        return "✅ Report finalized"
    if node == "input_guard" and update.get("status") == "blocked":
        return "⛔ Topic blocked by guardrails"
    return f"· {node}"


def run_steps(steps) -> None:
    """Consume a stream of (node, update) and show live progress."""
    with st.status("Working...", expanded=True) as status:
        try:
            for node, update in steps:
                st.session_state.report_log.append(describe(node, update))
                st.write(st.session_state.report_log[-1])
        except Exception as exc:  # network / API errors
            st.session_state.report_log.append(f"❌ {type(exc).__name__}: {exc}")
            st.error(st.session_state.report_log[-1])
        state = st.session_state.report_session.result["status"]
        status.update(label={"done": "Report ready", "awaiting_plan_review": "Waiting for plan review",
                             "blocked": "Blocked", "cancelled": "Cancelled"}.get(state, state),
                      state="complete" if state in ("done", "awaiting_plan_review") else "error", expanded=False)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📑 Research Reports")
    topic = st.text_area("Topic", "The current state of solid-state batteries for electric vehicles", height=90)
    audience = st.selectbox("Audience", ["technology-curious general readers", "executives", "engineers", "students"])
    review_plan = st.checkbox("Review the plan before researching", value=True)
    if st.button("Generate report", type="primary", use_container_width=True):
        st.session_state.report_session = ReportSession()
        st.session_state.report_log = []
        st.session_state.report_pending_run = (topic, audience, review_plan)
    st.caption(f"Budget: ≤{MAX_QUESTIONS} questions, 1 follow-up research round, 2 revisions.")

if "report_session" not in st.session_state:
    st.info("Enter a topic and click **Generate report**. Each report costs roughly $0.01 or less in API usage.")
    st.stop()

session: ReportSession = st.session_state.report_session

if run := st.session_state.pop("report_pending_run", None):
    run_steps(session.stream(run[0], audience=run[1], review_plan=run[2]))

# ---------------------------------------------------------------------------
# Plan review (human-in-the-loop)
# ---------------------------------------------------------------------------

if pending := session.pending_plan:
    st.subheader("📝 Review the research plan")
    st.caption("Edit, add or remove questions (one per line). Research starts only after you approve.")
    edited = st.text_area("Research questions", "\n".join(q["question"] for q in pending["plan"]), height=180)
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve and research", use_container_width=True):
        queries = {q["question"]: q["query"] for q in pending["plan"]}
        plan = [{"question": q, "query": queries.get(q, q)} for q in (line.strip() for line in edited.splitlines()) if q]
        run_steps(session.approve_plan(plan))
        st.rerun()
    if c2.button("❌ Cancel", use_container_width=True):
        run_steps(session.cancel_plan())
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

result = session.result
with st.expander("Progress log", expanded=False):
    for line in st.session_state.report_log:
        st.write(line)
for note in result["guard_notes"]:
    st.caption(f"🛡️ {note}")

if result["status"] == "done":
    report_col, info_col = st.columns([3, 1])
    with report_col:
        st.markdown(result["report"])
    with info_col:
        tokens = sum(u["input_tokens"] + u["output_tokens"] for u in result["usage"].values())
        st.metric("Review scores", " → ".join(map(str, result["scores"])) or "-")
        st.metric("Sources cited", len(result["references"]))
        st.metric("Revisions / research rounds", f"{result['revisions']} / {result['research_rounds']}")
        st.metric("Tokens", f"{tokens:,}", f"≈ ${estimate_cost(result['usage']):.4f}", delta_color="off")
        st.download_button("⬇️ Download Markdown", result["report"], file_name="report.md", use_container_width=True)
        if result["blocked_sources"]:
            with st.expander(f"🛡️ {len(result['blocked_sources'])} untrusted source(s) dropped"):
                for b in result["blocked_sources"]:
                    st.caption(f"{b['url']} — {b['reason']}")
        with st.expander("Reviewer feedback"):
            for i, r in enumerate(result["reviews"], 1):
                st.caption(f"**Review {i}: {r['score']}/10 → {r['decision']}**\n\n{r['feedback']}")
elif result["status"] == "cancelled":
    st.warning("Report cancelled at plan review; no research was run.")
elif result["status"] == "blocked":
    st.error("The topic was blocked by guardrails.")
