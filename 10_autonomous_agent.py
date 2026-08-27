"""
10 - Autonomous Agent
Pattern: Self-directed, long-running agent that autonomously selects and
         executes tools to accomplish a goal. Uses MemorySaver for
         persistence and a max_iterations guard to prevent runaway loops.
Use cases: Background research, automated report generation, long-horizon
           tasks, AI assistants that work independently without constant
           human prompting.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import Annotated, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def research_topic(topic: str) -> str:
    """Research a topic and return key findings."""
    mock_findings = {
        "quantum computing": (
            "Quantum computing uses qubits that can be in superposition. "
            "Key players: IBM, Google, IonQ. 2026 milestone: 1000+ qubit processors. "
            "Use cases: cryptography, drug discovery, optimization."
        ),
        "langchain": (
            "LangChain is a framework for building LLM applications. "
            "Key components: chains, agents, memory, tools. "
            "LangGraph is its graph-based agent orchestration layer."
        ),
        "ai trends": (
            "Top AI trends 2026: multimodal models, agentic AI, on-device inference, "
            "AI safety and alignment research, open-source model proliferation."
        ),
    }
    for key in mock_findings:
        if key in topic.lower():
            return mock_findings[key]
    return f"Research on '{topic}': Key insights include rapid adoption, emerging standards, and increasing investment in this domain."


@tool
def save_note(note: str) -> str:
    """Save an important note or insight for later use."""
    return f"Note saved: '{note[:100]}{'...' if len(note) > 100 else ''}'"


@tool
def generate_report(title: str, sections: str) -> str:
    """Generate a structured report with the given title and sections."""
    return (
        f"=== REPORT: {title} ===\n"
        f"Generated sections: {sections}\n"
        f"Status: Report compiled successfully with all requested information."
    )


@tool
def mark_goal_complete(summary: str) -> str:
    """Mark the current goal as complete with a summary of what was accomplished."""
    return f"GOAL_COMPLETE: {summary}"


TOOLS = [research_topic, save_note, generate_report, mark_goal_complete]
TOOL_MAP = {t.name: t for t in TOOLS}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AutonomousState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    goal: str
    progress_notes: List[str]
    iteration_count: int
    max_iterations: int
    completed: bool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)


def autonomous_agent_node(state: AutonomousState) -> dict:
    system = (
        f"You are an autonomous AI agent working to accomplish this goal:\n"
        f"GOAL: {state['goal']}\n\n"
        f"Progress so far:\n"
        + ("\n".join(f"  - {n}" for n in state["progress_notes"]) if state["progress_notes"] else "  None yet.")
        + f"\n\nIteration {state['iteration_count'] + 1} of {state['max_iterations']}. "
        + "Autonomously decide the next action. Use tools to make progress. "
        + "Call mark_goal_complete when the goal is fully accomplished."
    )
    resp = _llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
    return {"messages": [resp], "iteration_count": state["iteration_count"] + 1}


def tool_executor_node(state: AutonomousState) -> dict:
    last = state["messages"][-1]
    tool_results: List[ToolMessage] = []
    new_notes = list(state["progress_notes"])
    completed = state["completed"]

    for tc in last.tool_calls:
        output = TOOL_MAP[tc["name"]].invoke(tc["args"])
        tool_results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

        # Track progress notes from save_note calls
        if tc["name"] == "save_note":
            new_notes.append(tc["args"].get("note", "")[:100])

        # Detect goal completion
        if tc["name"] == "mark_goal_complete":
            completed = True

    return {"messages": tool_results, "progress_notes": new_notes, "completed": completed}


def autonomous_router(state: AutonomousState) -> str:
    if state["completed"] or state["iteration_count"] >= state["max_iterations"]:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "agent"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()

_builder = StateGraph(AutonomousState)
_builder.add_node("agent", autonomous_agent_node)
_builder.add_node("tools", tool_executor_node)
_builder.set_entry_point("agent")
_builder.add_conditional_edges("agent", autonomous_router, {"tools": "tools", "agent": "agent", END: END})
_builder.add_edge("tools", "agent")
app = _builder.compile(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 10 Autonomous Agent ===\n")
    goal = "Research quantum computing trends, save key insights, and generate a summary report."
    print(f"Goal: {goal}\n")

    config = {"configurable": {"thread_id": "autonomous-research-1"}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content=f"Begin working on this goal: {goal}")],
            "goal": goal,
            "progress_notes": [],
            "iteration_count": 0,
            "max_iterations": 6,
            "completed": False,
        },
        config=config,
    )

    print(f"Completed: {result['completed']}")
    print(f"Iterations used: {result['iteration_count']}")
    print(f"Progress notes: {len(result['progress_notes'])}")
    if result["progress_notes"]:
        for note in result["progress_notes"]:
            print(f"  - {note}")
    print(f"\nFinal message:\n{result['messages'][-1].content[:400]}")


if __name__ == "__main__":
    main()
