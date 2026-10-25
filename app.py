"""
Streamlit frontend for the 16 LangGraph agent modules.
Run: streamlit run app.py   (or double-click start.bat on Windows)

Each agent module is imported unchanged (they stay standalone-runnable);
this file only builds inputs, calls the module's compiled `app`, and renders results.
"""
import contextlib
import importlib.util
import io
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

ROOT = Path(__file__).parent
MODULE_FILES = sorted(p.name for p in ROOT.glob("[0-9][0-9]_*.py"))
TAVILY_MODULES = {"01", "02", "10"}

st.set_page_config(page_title="LangGraph Agents", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading agent module...")
def load_module(filename: str):
    """Import a module by path (names start with digits, so plain import won't work).
    Cached so MemorySaver checkpointers survive Streamlit reruns."""
    spec = importlib.util.spec_from_file_location(f"agent_{filename[:2]}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_key(num: str, name: str) -> str:
    return f"{num}_{name}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_messages(messages) -> None:
    """Show a message list: tool calls and tool results collapsed, final answer prominent."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**🧑 User:** {msg.content}")
        elif isinstance(msg, ToolMessage):
            with st.expander(f"🔧 Tool result: `{msg.name}`"):
                st.text(msg.content)
        elif isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", []) or []:
                st.caption(f"→ calls `{call['name']}` with `{call['args']}`")
            if msg.content:
                st.markdown(f"**🤖 Agent:** {msg.content}")


def run_demo(module) -> None:
    """Run the module's own main() and show what it prints."""
    buf = io.StringIO()
    with st.spinner("Running built-in demo..."):
        with contextlib.redirect_stdout(buf):
            module.main()
    st.code(buf.getvalue() or "(no output)", language="text")


def chat_ui(num: str, send) -> None:
    """Shared chat interface. `send(message, thread_id)` returns the reply text."""
    hist_key, thread_key = session_key(num, "history"), session_key(num, "thread")
    st.session_state.setdefault(hist_key, [])
    st.session_state.setdefault(thread_key, f"ui-{uuid.uuid4().hex[:8]}")

    if st.button("🔄 New conversation", key=session_key(num, "reset")):
        st.session_state[hist_key] = []
        st.session_state[thread_key] = f"ui-{uuid.uuid4().hex[:8]}"
        st.rerun()
    st.caption(f"Thread: `{st.session_state[thread_key]}`")

    for role, text in st.session_state[hist_key]:
        st.chat_message(role).write(text)

    if prompt := st.chat_input("Type a message..."):
        st.chat_message("user").write(prompt)
        with st.spinner("Thinking..."):
            reply = send(prompt, st.session_state[thread_key])
        st.chat_message("assistant").write(reply)
        st.session_state[hist_key] += [("user", prompt), ("assistant", reply)]


# ---------------------------------------------------------------------------
# Per-agent pages
# ---------------------------------------------------------------------------

def page_messages_agent(num, m, default):
    """01, 02: plain message-in, message-out tool agents."""
    query = st.text_area("Query", default, key=session_key(num, "q"))
    if st.button("Run", type="primary", key=session_key(num, "run")):
        with st.spinner("Running..."):
            result = m.app.invoke({"messages": [HumanMessage(content=query)]})
        render_messages(result["messages"])


def page_01(m):
    page_messages_agent("01", m, "What's the weather in Paris and London? Also what is 42 * 7?")


def page_02(m):
    page_messages_agent("02", m, "What's the stock price of AAPL and NVDA?")


def page_03(m):
    query = st.text_input("Query", "Write a haiku about the Python programming language.")
    if st.button("Run", type="primary"):
        with st.spinner("Routing..."):
            result = m.app.invoke({"query": query, "route": "", "response": ""})
        st.info(f"Routed to: **{result['route']}**")
        st.markdown(result["response"])


def page_04(m):
    objective = st.text_area(
        "Objective",
        "Research the main benefits of LangGraph for AI agent development and produce a 3-point summary.",
    )
    if st.button("Run", type="primary"):
        with st.spinner("Planning and executing..."):
            result = m.app.invoke({
                "objective": objective, "plan": [], "current_step": 0,
                "completed_steps": [], "final_answer": None,
            })
        with st.expander(f"Completed steps ({len(result['completed_steps'])})"):
            for step in result["completed_steps"]:
                st.markdown(f"- {step}")
        st.subheader("Final answer")
        st.markdown(result["final_answer"] or "(none)")


def page_05(m):
    task = st.text_area("Task", "Write a compelling 3-sentence product pitch for an AI-powered coding assistant.")
    max_iter = st.slider("Max iterations", 1, 5, 3)
    if st.button("Run", type="primary"):
        with st.spinner("Generating and critiquing..."):
            result = m.app.invoke({
                "task": task, "draft": "", "critique": "", "score": 0,
                "iteration": 0, "max_iterations": max_iter, "approved": False,
            })
        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations", result["iteration"])
        c2.metric("Score", f"{result['score']}/10")
        c3.metric("Approved", "Yes" if result["approved"] else "No")
        st.subheader("Final draft")
        st.markdown(result["draft"])
        if result["critique"]:
            with st.expander("Last critique"):
                st.write(result["critique"])


def page_06(m):
    user_id = st.text_input("User ID (long-term facts are stored per user)", "alice")
    chat_ui("06", lambda msg, thread: m.chat(user_id, thread, msg))


def page_07(m):
    from langgraph.types import Command

    stage_key, thread_key, proposed_key, result_key = (
        session_key("07", k) for k in ("stage", "thread", "proposed", "result")
    )
    st.session_state.setdefault(stage_key, "idle")

    task = st.text_input("Task", "Send a project status update email to the engineering team.")
    if st.button("Propose action", type="primary"):
        thread_id = f"hitl-{uuid.uuid4().hex[:8]}"
        with st.spinner("Planning..."):
            # First invoke runs until interrupt() in request_approval
            snapshot = m.app.invoke(
                {"task": task, "proposed_action": "", "approved": None, "result": ""},
                config={"configurable": {"thread_id": thread_id}},
            )
        st.session_state.update({
            stage_key: "pending", thread_key: thread_id,
            proposed_key: snapshot.get("proposed_action", "(unknown)"),
        })

    if st.session_state[stage_key] in ("pending", "done"):
        st.warning(f"**Agent proposes:** {st.session_state[proposed_key]}")

    if st.session_state[stage_key] == "pending":
        st.write("The graph is paused at `interrupt()`. Your decision resumes it:")
        c1, c2 = st.columns(2)
        decision = None
        if c1.button("✅ Approve", use_container_width=True):
            decision = True
        if c2.button("❌ Reject", use_container_width=True):
            decision = False
        if decision is not None:
            with st.spinner("Resuming..."):
                final = m.app.invoke(
                    Command(resume=decision),
                    config={"configurable": {"thread_id": st.session_state[thread_key]}},
                )
            st.session_state.update({stage_key: "done", result_key: final["result"]})
            st.rerun()

    if st.session_state[stage_key] == "done":
        st.success(st.session_state[result_key])


def page_08(m):
    objective = st.text_area("Objective", "Research LangGraph's key features and write a concise 2-paragraph overview.")
    if st.button("Run", type="primary"):
        with st.spinner("Supervisor delegating to workers..."):
            result = m.app.invoke({
                "messages": [HumanMessage(content=objective)], "objective": objective,
                "research_result": "", "written_result": "", "next_worker": "", "iteration": 0,
            })
        st.caption(f"Worker invocations: {result['iteration']}")
        with st.expander("Research findings"):
            st.markdown(result["research_result"])
        st.subheader("Written output")
        st.markdown(result["written_result"])


def page_09(m):
    query = st.text_input("Customer query", "I was charged twice on my last invoice. Can you help?")
    if st.button("Run", type="primary"):
        with st.spinner("Triaging..."):
            result = m.app.invoke({
                "messages": [HumanMessage(content=query)], "active_agent": "triage", "handoff_count": 0,
            })
        st.info(f"Handoffs: **{result['handoff_count']}** · Final agent: **{result['active_agent']}**")
        render_messages(result["messages"])


def page_10(m):
    goal = st.text_area("Goal", "Research quantum computing trends, save key insights, and generate a summary report.")
    max_iter = st.slider("Max iterations", 1, 10, 6)
    if st.button("Run", type="primary"):
        with st.spinner("Working autonomously..."):
            result = m.app.invoke(
                {
                    "messages": [HumanMessage(content=f"Begin working on this goal: {goal}")],
                    "goal": goal, "progress_notes": [], "iteration_count": 0,
                    "max_iterations": max_iter, "completed": False,
                },
                config={"configurable": {"thread_id": f"auto-{uuid.uuid4().hex[:8]}"}},
            )
        c1, c2, c3 = st.columns(3)
        c1.metric("Completed", "Yes" if result["completed"] else "No")
        c2.metric("Iterations", result["iteration_count"])
        c3.metric("Notes saved", len(result["progress_notes"]))
        for note in result["progress_notes"]:
            st.markdown(f"- {note}")
        with st.expander("Full trace"):
            render_messages(result["messages"])


def page_11(m):
    text = st.text_area("Email text", m.SAMPLE_EMAILS[0].strip(), height=220)
    if st.button("Extract", type="primary"):
        with st.spinner("Extracting..."):
            result = m.app.invoke({"raw_text": text, "analysis": None})
        st.json(result["analysis"])


def page_12(m):
    task = st.text_input("Task", "Generate and print the first 10 Fibonacci numbers.")
    if st.button("Run", type="primary"):
        with st.spinner("Writing and executing code..."):
            result = m.app.invoke({
                "messages": [HumanMessage(content=task)], "last_code": "",
                "last_output": "", "execution_count": 0,
            })
        st.caption(f"Executions: {result['execution_count']}")
        if result["last_code"]:
            st.code(result["last_code"], language="python")
        if result["last_output"]:
            st.text(result["last_output"])
        st.markdown(result["messages"][-1].content)


def page_13(m):
    items = st.text_input("Items (comma-separated)", "Python, Rust, TypeScript, Go")
    category = st.text_input("Analysis dimension", "ecosystem maturity and tooling")
    if st.button("Run", type="primary"):
        item_list = [i.strip() for i in items.split(",") if i.strip()]
        with st.spinner(f"Analyzing {len(item_list)} items in parallel..."):
            result = m.app.invoke({"items": item_list, "category": category, "analyses": [], "summary": ""})
        for analysis in result["analyses"]:
            with st.expander(analysis[:80]):
                st.write(analysis)
        st.subheader("Aggregated summary")
        st.markdown(result["summary"])


def page_14(m):
    query = st.text_input("Question", "How does the human-in-the-loop pattern work in LangGraph?")
    k = st.slider("Documents to retrieve", 1, 8, 4)
    if st.button("Ask", type="primary"):
        if m._vectorstore is None:
            with st.spinner("Building FAISS index (one-time, calls the embeddings API)..."):
                m._vectorstore = m._build_vectorstore()
        with st.spinner("Retrieving and generating..."):
            result = m.app.invoke({"query": query, "retrieved_docs": [], "answer": "", "num_results": k})
        with st.expander(f"Retrieved {len(result['retrieved_docs'])} documents"):
            for doc in result["retrieved_docs"]:
                st.markdown(f"- {doc}")
        st.markdown(result["answer"])


def page_15(m):
    task = st.text_area(
        "Task",
        "Write a product description for a smart water bottle that tracks hydration and syncs with a fitness app.",
    )
    c1, c2 = st.columns(2)
    target = c1.slider("Target score", 1, 10, 8)
    max_iter = c2.slider("Max iterations", 1, 6, 4)
    if st.button("Run", type="primary"):
        with st.spinner("Generating and evaluating..."):
            result = m.app.invoke({
                "task": task, "output": "", "score": 0, "feedback": "", "weaknesses": "",
                "iteration": 0, "max_iterations": max_iter, "target_score": target,
                "passed": False, "score_history": [],
            })
        c1, c2, c3 = st.columns(3)
        c1.metric("Final score", f"{result['score']}/10")
        c2.metric("Passed", "Yes" if result["passed"] else "No")
        c3.metric("Iterations", result["iteration"])
        if result["score_history"]:
            st.line_chart(result["score_history"])
        st.subheader("Final output")
        st.markdown(result["output"])


def page_16(m):
    chat_ui("16", lambda msg, thread: m.chat(msg, thread_id=thread))


PAGES = {f.split("_")[0]: globals()[f"page_{f.split('_')[0]}"] for f in MODULE_FILES}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 LangGraph Agents")
    choice = st.radio(
        "Agent",
        MODULE_FILES,
        format_func=lambda f: f"{f[:2]} · {f[3:-3].replace('_', ' ').title()}",
        label_visibility="collapsed",
    )
    st.divider()
    for var in ("OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"):
        st.caption(f"{'✅' if os.getenv(var) else '⚠️'} {var}")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.caption(f"LangSmith project: `{os.getenv('LANGCHAIN_PROJECT', 'default')}`")

num = choice[:2]
st.header(f"{num} · {choice[3:-3].replace('_', ' ').title()}")

if num in TAVILY_MODULES and not os.getenv("TAVILY_API_KEY"):
    st.error("This agent uses Tavily search. Add `TAVILY_API_KEY` to `.env` and restart.")
    st.stop()

try:
    module = load_module(choice)
except Exception as exc:  # surface import-time failures (missing keys, packages) in the UI
    st.error(f"Could not load `{choice}`: {exc}")
    st.stop()

if module.__doc__:
    with st.expander("About this pattern", expanded=False):
        st.text(module.__doc__.strip())

tab_run, tab_demo, tab_source = st.tabs(["▶ Run", "📜 Built-in demo", "🧾 Source"])
with tab_run:
    try:
        PAGES[num](module)
    except Exception as exc:  # e.g. invalid API key, rate limit — show it instead of a traceback
        st.error(f"{type(exc).__name__}: {exc}")
with tab_demo:
    st.caption(f"Runs `{choice}`'s own `main()` and shows its printed output.")
    if st.button("Run demo", key=f"demo_{num}"):
        try:
            run_demo(module)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
with tab_source:
    st.code((ROOT / choice).read_text(encoding="utf-8"), language="python")
