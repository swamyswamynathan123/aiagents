"""
Streamlit UI for the coding agent: pick a task, watch the agent work, review the diff.
Run from the repo root:  streamlit run projects/coding_agent/ui.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `projects.` and guardrails

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from projects.coding_agent import workspace as ws  # noqa: E402
from projects.coding_agent.graph import MAX_EDITS, MAX_STEPS, MODEL, CodingSession  # noqa: E402
from projects.coding_agent.tasks import TASKS, with_hidden_tests  # noqa: E402
from projects.research_report.graph import format_cost  # noqa: E402

st.set_page_config(page_title="Coding Agent", page_icon="🛠️", layout="wide")


def describe(node: str, update: dict) -> str:
    if node == "baseline":
        b = update["baseline"]
        return f"🧪 Baseline: {b['passed']} passed, {ws.failing_count(b)} failing"
    if node == "agent":
        calls = [c for m in update.get("messages", []) for c in (getattr(m, "tool_calls", None) or [])]
        return " · ".join(f"🔧 {c['name']}({c['args'].get('path') or c['args'].get('query') or c['args'].get('target') or ''})"
                          for c in calls)
    if node == "executor" and update.get("edits"):
        e = update["edits"][-1]
        icon = {"kept": "✏️", "reverted": "↩️", "rejected": "⛔"}[e["outcome"]]
        return f"{icon} edit {e['outcome']}: {e['path']} ({e['reason'][:80]})"
    if node == "verify" and "final_tests" in update:
        f = update["final_tests"]
        return f"✅ Verify: {f['passed']} passed, {ws.failing_count(f)} failing"
    return ""


with st.sidebar:
    st.title("🛠️ Coding Agent")
    task_id = st.selectbox("Task", list(TASKS), format_func=lambda t: TASKS[t]["title"])
    request = st.text_area("Request to the agent", TASKS[task_id]["request"], height=110)
    run = st.button("Run agent", type="primary", use_container_width=True)
    st.caption(f"Model `{MODEL}` · budget {MAX_STEPS} steps, {MAX_EDITS} edits. Test files are read-only; "
               "edits that make tests worse are reverted automatically.")

if run:
    session = CodingSession(task_id, request=request)
    st.session_state.coding_session = session
    with st.status("Working...", expanded=True) as status:
        try:
            for node, update in session.stream():
                if line := describe(node, update):
                    st.write(line)
        except Exception as exc:  # network / API errors
            st.error(f"{type(exc).__name__}: {exc}")
        r = session.result
        status.update(label={"fixed": "Fixed", "partial": "Partially fixed", "failed": "Not fixed",
                             "blocked": "Blocked"}.get(r["status"], r["status"]),
                      state="complete" if r["status"] == "fixed" else "error", expanded=False)
    st.session_state.coding_hidden = ws.run_tests(with_hidden_tests(session.workspace)) if r["diff"] else None

if "coding_session" not in st.session_state:
    st.info("Pick a task and click **Run agent**. Each task is a known bug in a small shop library "
            "(pricing, cart, inventory); the agent must make the failing tests pass without touching them.")
    st.stop()

r = st.session_state.coding_session.result
hidden = st.session_state.get("coding_hidden")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", r["status"])
c2.metric("Tests", f"{r['final_tests'].get('passed', 0)} passed", f"{ws.failing_count(r['final_tests']) if r['final_tests'] else '-'} failing",
          delta_color="off")
c3.metric("Hidden tests", "pass" if hidden and hidden["ok"] else "fail" if hidden else "-",
          help="Tests the agent never saw. They catch fixes that only special-case the visible tests.")
c4.metric("Steps / test runs", f"{r['steps']} / {r['test_runs']}")

st.subheader("Change summary")
st.write(r["summary"] or "-")
st.caption("Written from the diff, not from the agent's own account.")
if r["agent_summary"] and r["agent_summary"] != r["summary"]:
    with st.expander("What the agent said it did"):
        st.write(r["agent_summary"])
for e in r["rejected_test_edits"]:
    st.warning(f"🛡️ The agent tried to edit `{e['path']}`; rejected. Tests were not modified.")

st.subheader("Diff")
st.code(r["diff"] or "(no changes)", language="diff")
if r["diff"]:
    st.download_button("⬇️ Download patch", r["diff"], file_name=f"{task_id}.patch")

with st.expander(f"Edit history ({len(r['edits'])})"):
    st.table([{"file": e["path"], "outcome": e["outcome"], "detail": e["reason"][:120]} for e in r["edits"]] or [{}])
with st.expander("Final test output"):
    st.code(r["final_tests"].get("output", ""), language="text")
tokens = sum(u["input_tokens"] + u["output_tokens"] for u in r["usage"].values())
st.caption(f"Tokens: {tokens:,} ({format_cost(r['usage'])})")
