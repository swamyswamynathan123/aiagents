"""
Streamlit frontend for the 16 agent patterns, in two implementations:
  - LangGraph  (project root:  NN_*.py)
  - LangChain  (langchain/NN_*.py — LCEL chains + plain Python control flow)
Run: streamlit run app.py   (or double-click start.bat on Windows)

Agent modules are imported unchanged (they stay standalone-runnable).
Each page builds inputs and renders results; a framework-specific `run`
callable does the work: LangChain modules expose run(), LangGraph modules
are adapted below by calling their compiled `app`.
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
FRAMEWORK_DIRS = {"LangGraph": ROOT, "LangChain": ROOT / "langchain"}
TAVILY_MODULES = {"01", "02", "10"}

st.set_page_config(page_title="LangGraph Agents", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def module_files(framework: str):
    return sorted(p.name for p in FRAMEWORK_DIRS[framework].glob("[0-9][0-9]_*.py"))


@st.cache_resource(show_spinner="Loading agent module...")
def load_module(path: str):
    """Import a module by path (names start with digits, so plain import won't work).
    Cached so in-memory checkpointers and chat histories survive Streamlit reruns."""
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"agent_{p.parent.name}_{p.stem[:2]}", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_thread_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# LangGraph adapters — map page inputs onto each graph's state schema
# ---------------------------------------------------------------------------

def _lg_14(m, query, num_results):
    if m._vectorstore is None:
        m._vectorstore = m._build_vectorstore()
    return m.app.invoke({"query": query, "retrieved_docs": [], "answer": "", "num_results": num_results})


LANGGRAPH_RUN = {
    "01": lambda m, query: m.app.invoke({"messages": [HumanMessage(content=query)]}),
    "02": lambda m, query: m.app.invoke({"messages": [HumanMessage(content=query)]}),
    "03": lambda m, query: m.app.invoke({"query": query, "route": "", "response": ""}),
    "04": lambda m, objective: m.app.invoke({
        "objective": objective, "plan": [], "current_step": 0, "completed_steps": [], "final_answer": None,
    }),
    "05": lambda m, task, max_iterations: m.app.invoke({
        "task": task, "draft": "", "critique": "", "score": 0,
        "iteration": 0, "max_iterations": max_iterations, "approved": False,
    }),
    "08": lambda m, objective: m.app.invoke({
        "messages": [HumanMessage(content=objective)], "objective": objective,
        "research_result": "", "written_result": "", "next_worker": "", "iteration": 0,
    }),
    "09": lambda m, query: m.app.invoke({
        "messages": [HumanMessage(content=query)], "active_agent": "triage", "handoff_count": 0,
    }),
    "10": lambda m, goal, max_iterations: m.app.invoke(
        {
            "messages": [HumanMessage(content=f"Begin working on this goal: {goal}")],
            "goal": goal, "progress_notes": [], "iteration_count": 0,
            "max_iterations": max_iterations, "completed": False,
        },
        config={"configurable": {"thread_id": new_thread_id("auto")}},
    ),
    "11": lambda m, raw_text: m.app.invoke({"raw_text": raw_text, "analysis": None}),
    "12": lambda m, task: m.app.invoke({
        "messages": [HumanMessage(content=task)], "last_code": "", "last_output": "", "execution_count": 0,
    }),
    "13": lambda m, items, category: m.app.invoke({"items": items, "category": category, "analyses": [], "summary": ""}),
    "14": _lg_14,
    "15": lambda m, task, target_score, max_iterations: m.app.invoke({
        "task": task, "output": "", "score": 0, "feedback": "", "weaknesses": "",
        "iteration": 0, "max_iterations": max_iterations, "target_score": target_score,
        "passed": False, "score_history": [],
    }),
}


def make_runner(framework: str, num: str, m):
    if framework == "LangChain":
        return getattr(m, "run", None)
    adapter = LANGGRAPH_RUN.get(num)
    return (lambda **kw: adapter(m, **kw)) if adapter else None


def hitl_api(framework: str, m):
    """Human-in-the-loop (07) as two calls: propose(task) -> (token, action); resolve(token, approved) -> result."""
    if framework == "LangChain":
        return (lambda task: (None, m.propose_action(task))), (lambda action, token, approved: m.resolve(action, approved))

    from langgraph.types import Command

    def propose(task):
        thread_id = new_thread_id("hitl")
        # First invoke runs until interrupt() in request_approval
        snapshot = m.app.invoke(
            {"task": task, "proposed_action": "", "approved": None, "result": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        return thread_id, snapshot.get("proposed_action", "(unknown)")

    def resolve(action, thread_id, approved):
        final = m.app.invoke(Command(resume=approved), config={"configurable": {"thread_id": thread_id}})
        return final["result"]

    return propose, resolve


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


def chat_ui(key: str, send) -> None:
    """Shared chat interface. `send(message, thread_id)` returns the reply text."""
    hist_key, thread_key = f"{key}_history", f"{key}_thread"
    st.session_state.setdefault(hist_key, [])
    st.session_state.setdefault(thread_key, new_thread_id("ui"))

    if st.button("🔄 New conversation", key=f"{key}_reset"):
        st.session_state[hist_key] = []
        st.session_state[thread_key] = new_thread_id("ui")
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
# Per-agent pages — ctx carries: m (module), run (callable), key (session prefix), framework
# ---------------------------------------------------------------------------

def page_messages_agent(ctx, default):
    """01, 02: plain message-in, message-out tool agents."""
    query = st.text_area("Query", default, key=f"{ctx['key']}_q")
    if st.button("Run", type="primary", key=f"{ctx['key']}_run"):
        with st.spinner("Running..."):
            result = ctx["run"](query=query)
        render_messages(result["messages"])


def page_01(ctx):
    page_messages_agent(ctx, "What's the weather in Paris and London? Also what is 42 * 7?")


def page_02(ctx):
    page_messages_agent(ctx, "What's the stock price of AAPL and NVDA?")


def page_03(ctx):
    query = st.text_input("Query", "Write a haiku about the Python programming language.")
    if st.button("Run", type="primary"):
        with st.spinner("Routing..."):
            result = ctx["run"](query=query)
        st.info(f"Routed to: **{result['route']}**")
        st.markdown(result["response"])


def page_04(ctx):
    objective = st.text_area(
        "Objective",
        "Research the main benefits of LangGraph for AI agent development and produce a 3-point summary.",
    )
    if st.button("Run", type="primary"):
        with st.spinner("Planning and executing..."):
            result = ctx["run"](objective=objective)
        with st.expander(f"Completed steps ({len(result['completed_steps'])})"):
            for step in result["completed_steps"]:
                st.markdown(f"- {step}")
        st.subheader("Final answer")
        st.markdown(result["final_answer"] or "(none)")


def page_05(ctx):
    task = st.text_area("Task", "Write a compelling 3-sentence product pitch for an AI-powered coding assistant.")
    max_iter = st.slider("Max iterations", 1, 5, 3)
    if st.button("Run", type="primary"):
        with st.spinner("Generating and critiquing..."):
            result = ctx["run"](task=task, max_iterations=max_iter)
        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations", result["iteration"])
        c2.metric("Score", f"{result['score']}/10")
        c3.metric("Approved", "Yes" if result["approved"] else "No")
        st.subheader("Final draft")
        st.markdown(result["draft"])
        if result["critique"]:
            with st.expander("Last critique"):
                st.write(result["critique"])


def page_06(ctx):
    user_id = st.text_input("User ID (long-term facts are stored per user)", "alice")
    chat_ui(ctx["key"], lambda msg, thread: ctx["m"].chat(user_id, thread, msg))


def page_07(ctx):
    propose, resolve = hitl_api(ctx["framework"], ctx["m"])
    stage_key, token_key, proposed_key, result_key = (f"{ctx['key']}_{k}" for k in ("stage", "token", "proposed", "result"))
    st.session_state.setdefault(stage_key, "idle")

    task = st.text_input("Task", "Send a project status update email to the engineering team.")
    if st.button("Propose action", type="primary"):
        with st.spinner("Planning..."):
            token, proposed = propose(task)
        st.session_state.update({stage_key: "pending", token_key: token, proposed_key: proposed})

    if st.session_state[stage_key] in ("pending", "done"):
        st.warning(f"**Agent proposes:** {st.session_state[proposed_key]}")

    if st.session_state[stage_key] == "pending":
        if ctx["framework"] == "LangGraph":
            st.write("The graph is paused at `interrupt()`. Your decision resumes it:")
        else:
            st.write("The chain has returned; the app holds the proposal until you decide:")
        c1, c2 = st.columns(2)
        decision = None
        if c1.button("✅ Approve", use_container_width=True):
            decision = True
        if c2.button("❌ Reject", use_container_width=True):
            decision = False
        if decision is not None:
            with st.spinner("Resuming..."):
                result = resolve(st.session_state[proposed_key], st.session_state[token_key], decision)
            st.session_state.update({stage_key: "done", result_key: result})
            st.rerun()

    if st.session_state[stage_key] == "done":
        st.success(st.session_state[result_key])


def page_08(ctx):
    objective = st.text_area("Objective", "Research LangGraph's key features and write a concise 2-paragraph overview.")
    if st.button("Run", type="primary"):
        with st.spinner("Supervisor delegating to workers..."):
            result = ctx["run"](objective=objective)
        st.caption(f"Worker invocations: {result['iteration']}")
        with st.expander("Research findings"):
            st.markdown(result["research_result"])
        st.subheader("Written output")
        st.markdown(result["written_result"])


def page_09(ctx):
    query = st.text_input("Customer query", "I was charged twice on my last invoice. Can you help?")
    if st.button("Run", type="primary"):
        with st.spinner("Triaging..."):
            result = ctx["run"](query=query)
        st.info(f"Handoffs: **{result['handoff_count']}** · Final agent: **{result['active_agent']}**")
        render_messages(result["messages"])


def page_10(ctx):
    goal = st.text_area("Goal", "Research quantum computing trends, save key insights, and generate a summary report.")
    max_iter = st.slider("Max iterations", 1, 10, 6)
    if st.button("Run", type="primary"):
        with st.spinner("Working autonomously..."):
            result = ctx["run"](goal=goal, max_iterations=max_iter)
        c1, c2, c3 = st.columns(3)
        c1.metric("Completed", "Yes" if result["completed"] else "No")
        c2.metric("Iterations", result["iteration_count"])
        c3.metric("Notes saved", len(result["progress_notes"]))
        for note in result["progress_notes"]:
            st.markdown(f"- {note}")
        with st.expander("Full trace"):
            render_messages(result["messages"])


def page_11(ctx):
    text = st.text_area("Email text", ctx["m"].SAMPLE_EMAILS[0].strip(), height=220)
    if st.button("Extract", type="primary"):
        with st.spinner("Extracting..."):
            result = ctx["run"](raw_text=text)
        st.json(result["analysis"])


def page_12(ctx):
    task = st.text_input("Task", "Generate and print the first 10 Fibonacci numbers.")
    if st.button("Run", type="primary"):
        with st.spinner("Writing and executing code..."):
            result = ctx["run"](task=task)
        st.caption(f"Executions: {result['execution_count']}")
        if result["last_code"]:
            st.code(result["last_code"], language="python")
        if result["last_output"]:
            st.text(result["last_output"])
        st.markdown(result["messages"][-1].content)


def page_13(ctx):
    items = st.text_input("Items (comma-separated)", "Python, Rust, TypeScript, Go")
    category = st.text_input("Analysis dimension", "ecosystem maturity and tooling")
    if st.button("Run", type="primary"):
        item_list = [i.strip() for i in items.split(",") if i.strip()]
        with st.spinner(f"Analyzing {len(item_list)} items in parallel..."):
            result = ctx["run"](items=item_list, category=category)
        for analysis in result["analyses"]:
            with st.expander(analysis[:80]):
                st.write(analysis)
        st.subheader("Aggregated summary")
        st.markdown(result["summary"])


def page_14(ctx):
    query = st.text_input("Question", "How does the human-in-the-loop pattern work in LangGraph?")
    k = st.slider("Documents to retrieve", 1, 8, 4)
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving and generating (first run builds the FAISS index)..."):
            result = ctx["run"](query=query, num_results=k)
        with st.expander(f"Retrieved {len(result['retrieved_docs'])} documents"):
            for doc in result["retrieved_docs"]:
                st.markdown(f"- {doc}")
        st.markdown(result["answer"])


def page_15(ctx):
    task = st.text_area(
        "Task",
        "Write a product description for a smart water bottle that tracks hydration and syncs with a fitness app.",
    )
    c1, c2 = st.columns(2)
    target = c1.slider("Target score", 1, 10, 8)
    max_iter = c2.slider("Max iterations", 1, 6, 4)
    if st.button("Run", type="primary"):
        with st.spinner("Generating and evaluating..."):
            result = ctx["run"](task=task, target_score=target, max_iterations=max_iter)
        c1, c2, c3 = st.columns(3)
        c1.metric("Final score", f"{result['score']}/10")
        c2.metric("Passed", "Yes" if result["passed"] else "No")
        c3.metric("Iterations", result["iteration"])
        if result["score_history"]:
            st.line_chart(result["score_history"])
        st.subheader("Final output")
        st.markdown(result["output"])


def page_16(ctx):
    chat_ui(ctx["key"], lambda msg, thread: ctx["m"].chat(msg, thread_id=thread))


PAGES = {f"{n:02d}": globals()[f"page_{n:02d}"] for n in range(1, 17)}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 Agent Patterns")
    framework = st.radio("Framework", list(FRAMEWORK_DIRS), horizontal=True)
    st.caption(
        "StateGraph nodes, edges and checkpointers." if framework == "LangGraph"
        else "LCEL chains with plain Python loops; no LangGraph."
    )
    choice = st.radio(
        "Agent",
        module_files(framework),
        format_func=lambda f: f"{f[:2]} · {f[3:-3].replace('_', ' ').title()}",
        label_visibility="collapsed",
    )
    st.divider()
    for var in ("OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"):
        st.caption(f"{'✅' if os.getenv(var) else '⚠️'} {var}")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.caption(f"LangSmith project: `{os.getenv('LANGCHAIN_PROJECT', 'default')}`")

num = choice[:2]
path = FRAMEWORK_DIRS[framework] / choice
st.header(f"{num} · {choice[3:-3].replace('_', ' ').title()}")
st.caption(f"{framework} · `{path.relative_to(ROOT).as_posix()}`")

if num in TAVILY_MODULES and not os.getenv("TAVILY_API_KEY"):
    st.error("This agent uses Tavily search. Add `TAVILY_API_KEY` to `.env` and restart.")
    st.stop()

try:
    module = load_module(str(path))
except Exception as exc:  # surface import-time failures (missing keys, packages) in the UI
    st.error(f"Could not load `{choice}`: {exc}")
    st.stop()

if module.__doc__:
    with st.expander("About this pattern", expanded=False):
        st.text(module.__doc__.strip())

ctx = {"m": module, "run": make_runner(framework, num, module), "key": f"{framework}_{num}", "framework": framework}

tab_run, tab_demo, tab_source = st.tabs(["▶ Run", "📜 Built-in demo", "🧾 Source"])
with tab_run:
    try:
        PAGES[num](ctx)
    except Exception as exc:  # e.g. invalid API key, rate limit — show it instead of a traceback
        st.error(f"{type(exc).__name__}: {exc}")
with tab_demo:
    st.caption(f"Runs `{choice}`'s own `main()` and shows its printed output.")
    if st.button("Run demo", key=f"demo_{ctx['key']}"):
        try:
            run_demo(module)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
with tab_source:
    st.code(path.read_text(encoding="utf-8"), language="python")
