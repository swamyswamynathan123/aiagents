"""
Coding agent — the LangGraph graph.

    input_guard ─ blocked ─► END
        │
     baseline (code: run the tests, record what fails)
        │
     agent ⇄ executor      the model reads, searches, edits and runs tests via tools;
        │                  after every edit the executor reruns the tests and REVERTS the edit
        │                  if more tests fail than before (budgets: steps and edits)
        ▼
     verify (code: full test run) ── failures & retry left ─► agent
        │
     finalize (diff of the agent's changes, status) ─► END

Patterns: tool loop (01), autonomous agent (10), code execution (12), reflection (05) with the
test suite as the verifier, planning (04: the agent is asked to state a hypothesis first), guardrails.
"""
import os
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

from guardrails import Guard, GuardrailBlocked, GuardrailConfig

from . import workspace as ws
from .tasks import TASKS, make_workspace

MAX_STEPS = 25     # agent LLM calls per task
MAX_EDITS = 12     # successful edits per task
MAX_VERIFY_RETRIES = 1
GUARD = GuardrailConfig(max_input_chars=2000, pii=False)  # code requests contain no personal data

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CodingState(TypedDict, total=False):
    request: str
    workspace: str
    original: dict            # file snapshot before the agent started (for the diff)
    messages: Annotated[List[BaseMessage], add_messages]
    baseline: dict            # test result before any change
    best_failing: int         # fewest failing tests reached so far (the rollback bar) ...
    best_passed: int          # ... and most passing (a syntax error makes tests error, not fail)
    stalled_edits: int        # consecutive edits that didn't reduce failures (thrash detection)
    edits: List[dict]         # every attempted edit: path, kept/reverted/rejected, reason
    test_runs: int
    steps: int
    verify_retries: int
    final_tests: dict
    diff: str
    summary: str              # written from the diff (trustworthy)
    agent_summary: str        # what the agent said it did (may be wrong)
    status: str               # blocked | fixed | partial | failed

# ---------------------------------------------------------------------------
# Tools (schemas only; the executor node runs them so it can enforce the rules)
# ---------------------------------------------------------------------------

@tool
def list_files() -> str:
    """List every file in the repository."""


@tool
def read_file(path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    """Read a file with line numbers (up to 250 lines per call)."""


@tool
def search_code(query: str) -> str:
    """Find lines containing the text (case-insensitive) across all Python files."""


@tool
def run_tests(target: Optional[str] = None) -> str:
    """Run the test suite, or one file / test id such as tests/test_cart.py::test_subtotal."""


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact occurrence of old_text with new_text. old_text must match exactly once,
    including indentation; copy it from read_file output without the line numbers. Test files
    are read-only. The tests rerun automatically after each edit."""


@tool
def finish(summary: str) -> str:
    """Call when the tests pass: summarize the root cause and the fix in 1-3 sentences."""


TOOLS = [list_files, read_file, search_code, run_tests, edit_file, finish]

SYSTEM_PROMPT = """You are a careful software engineer fixing a Python repository.

Work like this:
1. Run the tests and read the failures.
2. Read the code under test and state your hypothesis about the root cause before editing.
3. Make the smallest change that fixes the root cause. Fix the code, never the tests, and don't
   special-case test inputs: the fix must be correct in general.
4. The tests rerun after every edit. If an edit makes things worse, it is reverted automatically;
   rethink rather than retrying the same change.
5. When all tests pass, call finish with a short summary of the root cause and fix.

Budget: {steps} steps and {edits} edits."""

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model
_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)
_reporter = ChatOpenAI(model=MODEL, temperature=0)

REPORT_PROMPT = ("Write a 1-3 sentence change summary for a code reviewer. Describe ONLY what the diff "
                 "actually changes and the root cause it addresses. Do not mention anything that isn't in the diff.")

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def input_guard(state: CodingState) -> Command:
    guard = Guard(GUARD)
    try:
        request = guard.check_input(state["request"])
    except GuardrailBlocked as exc:
        return Command(goto=END, update={"status": "blocked", "summary": "; ".join(
            f"{v.check}: {v.detail}" for v in exc.violations)})
    return Command(goto="baseline", update={"request": request})


def baseline(state: CodingState) -> dict:
    root = Path(state["workspace"])
    result = ws.run_tests(root)
    system = SYSTEM_PROMPT.format(steps=MAX_STEPS, edits=MAX_EDITS)
    return {
        "original": ws.snapshot(root), "baseline": result, "best_failing": ws.failing_count(result),
        "best_passed": result["passed"], "stalled_edits": 0,
        "edits": [], "test_runs": 1, "steps": 0, "verify_retries": 0,
        "messages": [SystemMessage(content=system), HumanMessage(content=state["request"])],
    }


def agent(state: CodingState) -> Command:
    if state["steps"] >= MAX_STEPS:
        return Command(goto="verify")
    response = _llm.invoke(state["messages"])
    update = {"messages": [response], "steps": state["steps"] + 1}
    if not response.tool_calls:  # stopped without finish(): let verification decide
        return Command(goto="verify", update={**update, "summary": response.content})
    return Command(goto="executor", update=update)


def executor(state: CodingState) -> Command:
    root = Path(state["workspace"])
    edits, results = list(state["edits"]), []
    best, best_passed, runs, summary = state["best_failing"], state["best_passed"], state["test_runs"], None
    stalled = state["stalled_edits"]

    for call in state["messages"][-1].tool_calls:
        name, args = call["name"], call["args"]
        try:
            if name == "list_files":
                content = ws.list_files(root)
            elif name == "read_file":
                content = ws.read_file(root, args["path"], args.get("start_line") or 1, args.get("end_line"))
            elif name == "search_code":
                content = ws.search_code(root, args["query"])
            elif name == "run_tests":
                result, runs = ws.run_tests(root, args.get("target")), runs + 1
                content = result["output"]
            elif name == "edit_file":
                if sum(e["outcome"] == "kept" for e in edits) >= MAX_EDITS:
                    raise ws.ToolError("Edit budget used up. Call finish, or explain what is still failing.")
                signature = (args["path"], args["old_text"], args["new_text"])
                if any(e.get("signature") == signature for e in edits if e["outcome"] == "reverted"):
                    raise ws.ToolError("This exact edit was already reverted because it made things worse. "
                                       "Try a different fix.")
                previous = ws.edit_file(root, args["path"], args["old_text"], args["new_text"])
                result, runs = ws.run_tests(root), runs + 1
                now = ws.failing_count(result)
                # Regression = more failures OR fewer passes (a syntax error turns failures into
                # collection errors without increasing the failure count)
                if now > best or result["passed"] < best_passed:
                    (root / args["path"]).write_text(previous, encoding="utf-8", newline="\n")
                    edits.append({"path": args["path"], "outcome": "reverted", "signature": signature,
                                  "reason": f"{best} → {now} failing, {best_passed} → {result['passed']} passing"})
                    content = (f"Edit REVERTED: failing tests {best} → {now}, passing {best_passed} → "
                               f"{result['passed']}.\n{result['output'][-1500:]}")
                    stalled += 1
                else:
                    stalled = 0 if now < best else stalled + 1
                    best, best_passed = now, result["passed"]
                    edits.append({"path": args["path"], "outcome": "kept", "reason": f"{now} failing"})
                    # Show the edited lines so the model works from the file as it now is
                    content = (f"Edit applied. Tests now: {result['passed']} passed, {now} failing.\n"
                               f"{ws.excerpt(root, args['path'], args['new_text'])}\n"
                               + (result["output"][-1200:] if now else "All tests pass."))
                if stalled >= 3:
                    content += ("\n\nYou've made 3 edits without reducing the failures. Stop editing: re-read the "
                                "code under test from the top, restate your hypothesis, then make one targeted change.")
                    stalled = 0
            elif name == "finish":
                summary, content = args.get("summary", ""), "Finishing."
            else:
                content = f"Unknown tool {name}."
        except ws.ToolError as exc:
            if name == "edit_file":
                edits.append({"path": args.get("path", "?"), "outcome": "rejected", "reason": str(exc)})
                stalled += 1
            content = f"Error: {exc}"
        except (KeyError, TypeError) as exc:  # missing or malformed arguments
            content = f"Error: bad arguments for {name}: {exc}"
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))

    update = {"messages": results, "edits": edits, "best_failing": best, "best_passed": best_passed,
              "stalled_edits": stalled, "test_runs": runs}
    if summary is not None:
        return Command(goto="verify", update={**update, "summary": summary})
    return Command(goto="agent", update=update)


def verify(state: CodingState) -> Command:
    """The test suite is the verifier: no LLM judgement involved."""
    result = ws.run_tests(Path(state["workspace"]))
    runs = state["test_runs"] + 1
    if not result["ok"] and state["verify_retries"] < MAX_VERIFY_RETRIES and state["steps"] < MAX_STEPS:
        feedback = ("You finished, but the full test run still fails:\n" + result["output"][-2000:]
                    + "\nKeep going: fix the remaining failures, then call finish again.")
        return Command(goto="agent", update={"verify_retries": state["verify_retries"] + 1, "test_runs": runs,
                                             "messages": [HumanMessage(content=feedback)]})
    return Command(goto="finalize", update={"final_tests": result, "test_runs": runs})


def finalize(state: CodingState) -> dict:
    root = Path(state["workspace"])
    diff = ws.unified_diff(state["original"], ws.snapshot(root))
    final, base = state["final_tests"], state["baseline"]
    if final["ok"]:
        status = "fixed"
    elif ws.failing_count(final) < ws.failing_count(base):
        status = "partial"
    else:
        status = "failed"
    # The agent's own narrative can claim things that didn't happen (e.g. "I updated the test" when
    # that edit was rejected), so the summary users see is written from the diff alone.
    if diff:
        summary = _reporter.invoke([
            SystemMessage(content=REPORT_PROMPT),
            HumanMessage(content=f"Request: {state['request']}\n\nTests after the change: {final['passed']} passed, "
                                 f"{ws.failing_count(final)} failing\n\nDiff:\n{diff[:6000]}"),
        ]).content.strip()
    else:
        summary = "No changes were made."
    return {"diff": diff, "status": status, "summary": summary, "agent_summary": state.get("summary") or ""}

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_app(checkpointer=None):
    builder = StateGraph(CodingState)
    for name, fn in [("input_guard", input_guard), ("baseline", baseline), ("agent", agent),
                     ("executor", executor), ("verify", verify), ("finalize", finalize)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "input_guard")
    builder.add_edge("baseline", "agent")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


app = build_app()

# ---------------------------------------------------------------------------
# Session API (CLI demo, UI, evals, tests)
# ---------------------------------------------------------------------------

class CodingSession:
    """One task in its own workspace (a copy of the sample repo with the task's bug)."""

    def __init__(self, task_id: str, request: Optional[str] = None, graph=None):
        assert task_id in TASKS, task_id
        self.task_id = task_id
        self.request = request or TASKS[task_id]["request"]
        self.workspace = make_workspace(task_id)
        self.graph = graph or app
        self.config = {"configurable": {"thread_id": f"coding-{uuid.uuid4().hex[:8]}"},
                       "recursion_limit": 4 * MAX_STEPS + 20}
        self.usage: dict = {}

    def stream(self):
        from langchain_core.callbacks import get_usage_metadata_callback

        with get_usage_metadata_callback() as cb:
            for chunk in self.graph.stream({"request": self.request, "workspace": str(self.workspace)},
                                           self.config, stream_mode="updates"):
                yield from ((node, update or {}) for node, update in chunk.items())
        for model, u in cb.usage_metadata.items():
            totals = self.usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            totals["input_tokens"] += u.get("input_tokens", 0)
            totals["output_tokens"] += u.get("output_tokens", 0)

    def run(self) -> dict:
        for _ in self.stream():
            pass
        return self.result

    @property
    def result(self) -> dict:
        s = self.graph.get_state(self.config).values
        tool_calls = [c["name"] for m in s.get("messages", []) if isinstance(m, AIMessage) for c in m.tool_calls or []]
        return {
            "status": s.get("status", "running"), "summary": s.get("summary", ""), "diff": s.get("diff", ""),
            "agent_summary": s.get("agent_summary", ""),
            "rejected_test_edits": [e for e in s.get("edits", []) if e["outcome"] == "rejected" and ws.is_test_file(e["path"])],
            "baseline": s.get("baseline", {}), "final_tests": s.get("final_tests", {}),
            "edits": s.get("edits", []), "steps": s.get("steps", 0), "test_runs": s.get("test_runs", 0),
            "tool_calls": tool_calls, "workspace": str(self.workspace), "usage": self.usage,
        }
