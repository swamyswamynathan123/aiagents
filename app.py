"""
Streamlit frontend for the 16 agent patterns, in two implementations:
  - LangGraph  (project root:  NN_*.py)
  - LangChain  (langchain/NN_*.py — LCEL chains + plain Python control flow)
Run: streamlit run app.py   (or double-click start.bat on Windows)

Agent modules are imported unchanged (they stay standalone-runnable).
Each page builds inputs and renders results; a framework-specific `run`
callable from registry.py does the work. When guardrails are on, every
agent call goes through guardrails.Guard (input checks + output redaction).
"""
import contextlib
import io
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import registry
from guardrails import Guard, GuardrailBlocked, GuardrailConfig
from registry import FRAMEWORK_DIRS, ROOT, TAVILY_MODULES, hitl_api, make_runner, module_files, new_thread_id

load_dotenv()

# Agents whose job needs the PII the guardrail would redact
PII_EXEMPT = {"11": "extracting contact details is this agent's job"}

st.set_page_config(page_title="LangGraph Agents", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading agent module...")
def load_module(path: str):
    """Cached so in-memory checkpointers and chat histories survive Streamlit reruns."""
    return registry.load_module(path)


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


def guarded_send(guard, send):
    """Wrap a chat `send(message, thread_id)` with input checks and output redaction."""
    if guard is None:
        return send
    return lambda msg, thread: guard.check_output(send(guard.check_input(msg), thread))


def render_guard_log(guard) -> None:
    for stage, v in guard.log if guard else []:
        icon = {"block": "⛔", "redact": "✂️", "warn": "⚠️"}[v.action]
        st.caption(f"🛡️ {icon} {stage} · {v.check}: {v.detail}")


def render_eval_rows(rows) -> None:
    passed = sum(r["passed"] for r in rows)
    (st.success if passed == len(rows) else st.error)(f"{passed}/{len(rows)} examples passed")
    for r in rows:
        label = ", ".join(f"{k}={str(v)[:40]}" for k, v in r["inputs"].items())
        with st.expander(f"{'✅' if r['passed'] else '❌'} {label}"):
            st.table([{"check": k, "pass": "✅" if s else "❌", "detail": c or ""} for k, s, c in r["checks"]])
            st.caption("Agent answer")
            st.write(r["outputs"].get("answer") or r["outputs"].get("error") or "(empty)")


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
    chat_ui(ctx["key"], guarded_send(ctx["guard"], lambda msg, thread: ctx["m"].chat(user_id, thread, msg)))


def page_07(ctx):
    propose, resolve = hitl_api(ctx["framework"], ctx["m"])
    if guard := ctx["guard"]:
        _propose, _resolve = propose, resolve
        propose = lambda task: guard.check_output(_propose(guard.check_input(task)))  # noqa: E731
        resolve = lambda action, token, ok: guard.check_output(_resolve(action, token, ok))  # noqa: E731
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
    chat_ui(ctx["key"], guarded_send(ctx["guard"], lambda msg, thread: ctx["m"].chat(msg, thread_id=thread)))


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
    with st.expander("🛡️ Guardrails", expanded=False):
        guard_on = st.toggle("Enable guardrails", value=True)
        g_moderation = st.checkbox("Content moderation (OpenAI)", value=True, disabled=not guard_on)
        g_injection = st.checkbox("Prompt-injection heuristics", value=True, disabled=not guard_on)
        g_injection_llm = st.checkbox("LLM injection classifier (+1 call)", value=False, disabled=not guard_on)
        g_pii = st.checkbox("PII redaction", value=True, disabled=not guard_on)
        g_max = st.number_input("Max input chars", 100, 20000, 4000, step=500, disabled=not guard_on)
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

guard = None
if guard_on:
    pii = g_pii and num not in PII_EXEMPT
    guard = Guard(GuardrailConfig(
        moderation=g_moderation, injection=g_injection, injection_llm=g_injection_llm, pii=pii, max_input_chars=int(g_max),
    ))
    if g_pii and not pii:
        st.caption(f"🛡️ PII redaction is off for this agent: {PII_EXEMPT[num]}.")

runner = make_runner(framework, num, module)
ctx = {
    "m": module, "key": f"{framework}_{num}", "framework": framework, "guard": guard,
    "run": guard.wrap(runner) if guard and runner else runner,
}

tab_run, tab_demo, tab_evals, tab_source = st.tabs(["▶ Run", "📜 Built-in demo", "🧪 Evals", "🧾 Source"])
with tab_run:
    try:
        PAGES[num](ctx)
    except GuardrailBlocked as blocked:
        st.error(f"🛡️ Blocked by guardrails ({blocked.stage}): "
                 + "; ".join(f"**{v.check}** — {v.detail}" for v in blocked.violations))
    except Exception as exc:  # e.g. invalid API key, rate limit — show it instead of a traceback
        st.error(f"{type(exc).__name__}: {exc}")
    render_guard_log(guard)
with tab_demo:
    st.caption(f"Runs `{choice}`'s own `main()` and shows its printed output.")
    if st.button("Run demo", key=f"demo_{ctx['key']}"):
        try:
            run_demo(module)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
with tab_evals:
    st.caption(f"Runs this agent's cases from `evals/cases.py` against the {framework} version "
               "(guardrails are not applied). Full suite: `python evals/run_evals.py`.")
    c1, c2 = st.columns(2)
    use_judge = c1.checkbox("LLM-as-judge checks", value=True, key=f"judge_{ctx['key']}")
    upload = c2.checkbox("Log experiment to LangSmith", value=False, key=f"upload_{ctx['key']}",
                         disabled=not os.getenv("LANGCHAIN_API_KEY"))
    if st.button("Run evals", key=f"evals_{ctx['key']}"):
        from evals.run_evals import run_agent_evals
        try:
            with st.spinner("Running eval cases..."):
                render_eval_rows(run_agent_evals(framework, num, module, upload=upload, use_judge=use_judge))
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
with tab_source:
    st.code(path.read_text(encoding="utf-8"), language="python")
