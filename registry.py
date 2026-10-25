"""
Agent registry shared by the Streamlit UI (app.py) and the eval runner (evals/).

Loads agent modules by path and gives every agent a uniform calling interface:
  - make_runner(framework, num, module) -> run(**inputs) -> result dict
    (LangChain modules expose run(); LangGraph modules are adapted here by
    mapping inputs onto each graph's state schema)
  - hitl_api(framework, module) -> (propose, resolve) for the 07 approval flow
Chat agents (06, 16) are called through their module's chat() directly.
"""
import importlib.util
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

ROOT = Path(__file__).parent
FRAMEWORK_DIRS = {"LangGraph": ROOT, "LangChain": ROOT / "langchain"}
TAVILY_MODULES = {"01", "02", "10"}


def module_files(framework: str):
    return sorted(p.name for p in FRAMEWORK_DIRS[framework].glob("[0-9][0-9]_*.py"))


def module_path(framework: str, num: str) -> Path:
    return next(FRAMEWORK_DIRS[framework].glob(f"{num}_*.py"))


def load_module(path):
    """Import a module by path (names start with digits, so plain import won't work)."""
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"agent_{p.parent.name}_{p.stem[:2]}", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_thread_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# LangGraph adapters — map inputs onto each graph's state schema
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
    """Human-in-the-loop (07) as two calls: propose(task) -> (token, action); resolve(action, token, approved) -> result."""
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
