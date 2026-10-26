"""
10 - Autonomous Agent (LangChain)
Pattern: Self-directed tool loop — the LLM picks tools toward a goal, the
         loop records progress notes and stops on mark_goal_complete or
         when max_iterations is reached.
Use cases: Background research, automated report generation, long-horizon
           tasks, AI assistants that work independently without constant
           human prompting.
LangGraph equivalent: ../10_autonomous_agent.py (agent ⇄ tools graph + MemorySaver)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Tavily web search — requires TAVILY_API_KEY in .env
_tavily = TavilySearch(max_results=5, include_answer=True)


def _format_results(response) -> str:
    """Flatten a Tavily response into plain text the LLM can read."""
    if not isinstance(response, dict):
        return str(response)  # handled tool error, e.g. no results
    if "error" in response:
        return f"Search failed: {response['error']}"
    lines = [f"Answer: {response['answer']}"] if response.get("answer") else []
    for r in response.get("results", []):
        lines.append(f"- {r['title']} ({r['url']}): {r['content'][:300]}")
    return "\n".join(lines) or "No results found."


@tool
def research_topic(topic: str) -> str:
    """Research a topic and return key findings."""
    return _format_results(_tavily.invoke({"query": topic}))


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

_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# Autonomous loop
# ---------------------------------------------------------------------------

def _system_prompt(goal: str, notes: List[str], iteration: int, max_iterations: int) -> str:
    return (
        f"You are an autonomous AI agent working to accomplish this goal:\n"
        f"GOAL: {goal}\n\n"
        f"Progress so far:\n"
        + ("\n".join(f"  - {n}" for n in notes) if notes else "  None yet.")
        + f"\n\nIteration {iteration + 1} of {max_iterations}. "
        + "Autonomously decide the next action. Use tools to make progress. "
        + "Call mark_goal_complete when the goal is fully accomplished."
    )


def run(goal: str, max_iterations: int = 6) -> dict:
    messages: List[BaseMessage] = [HumanMessage(content=f"Begin working on this goal: {goal}")]
    notes: List[str] = []
    completed, iteration = False, 0

    while not completed and iteration < max_iterations:
        response = _llm.invoke([SystemMessage(content=_system_prompt(goal, notes, iteration, max_iterations))] + messages)
        messages.append(response)
        iteration += 1

        for tc in response.tool_calls:
            output = TOOL_MAP[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
            if tc["name"] == "save_note":
                notes.append(tc["args"].get("note", "")[:100])
            if tc["name"] == "mark_goal_complete":
                completed = True

    return {"messages": messages, "progress_notes": notes, "iteration_count": iteration, "completed": completed}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 10 Autonomous Agent (LangChain) ===\n")
    goal = "Research quantum computing trends, save key insights, and generate a summary report."
    print(f"Goal: {goal}\n")

    result = run(goal, max_iterations=6)

    print(f"Completed: {result['completed']}")
    print(f"Iterations used: {result['iteration_count']}")
    print(f"Progress notes: {len(result['progress_notes'])}")
    for note in result["progress_notes"]:
        print(f"  - {note}")
    print(f"\nFinal message:\n{result['messages'][-1].content[:400]}")


if __name__ == "__main__":
    main()
