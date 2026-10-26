"""
07 - Human-in-the-Loop Agent
Pattern: interrupt() pauses graph execution at a node and surfaces a decision
         to the human. Command(resume=value) continues execution with the
         human's response.
Use cases: Approval workflows (send email, deploy code, delete data),
           content review before publishing, sensitive operations requiring
           human oversight, compliance checkpoints.

How it works:
  1. First invoke() runs until interrupt() is hit — graph pauses and returns.
  2. Human reviews the proposed action.
  3. Second invoke(Command(resume=<decision>), config) resumes from the
     interrupt point. interrupt() returns the human's decision value.

Requires: MemorySaver (state must persist across the interrupt boundary).

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ApprovalState(TypedDict):
    task: str
    proposed_action: str
    approved: Optional[bool]
    result: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_action(state: ApprovalState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content=(
            "You are an AI assistant that takes actions on behalf of users. "
            "Describe the specific action you would take to accomplish the task. "
            "Be concise and specific (1-2 sentences)."
        )),
        HumanMessage(content=f"Task: {state['task']}"),
    ])
    return {"proposed_action": resp.content}


def request_approval(state: ApprovalState) -> dict:
    # interrupt() halts graph execution and surfaces payload to the caller.
    # When resumed with Command(resume=value), interrupt() returns that value.
    human_decision = interrupt({
        "question": "Do you approve this action?",
        "proposed_action": state["proposed_action"],
        "instructions": "Resume with True to approve, False to reject.",
    })
    return {"approved": bool(human_decision)}


def execute_action(state: ApprovalState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content="Execute the following action and confirm completion."),
        HumanMessage(content=f"Action: {state['proposed_action']}"),
    ])
    return {"result": f"[EXECUTED] {resp.content}"}


def skip_action(state: ApprovalState) -> dict:
    return {"result": f"[CANCELLED] Action rejected by human: '{state['proposed_action']}'"}


def approval_router(state: ApprovalState) -> str:
    return "execute" if state.get("approved") else "skip"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()

_builder = StateGraph(ApprovalState)
_builder.add_node("plan", plan_action)
_builder.add_node("approval", request_approval)
_builder.add_node("execute", execute_action)
_builder.add_node("skip", skip_action)
_builder.set_entry_point("plan")
_builder.add_edge("plan", "approval")
_builder.add_conditional_edges("approval", approval_router, {"execute": "execute", "skip": "skip"})
_builder.add_edge("execute", END)
_builder.add_edge("skip", END)
app = _builder.compile(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# Demo — two invocations demonstrating the interrupt/resume cycle
# ---------------------------------------------------------------------------

def run_with_approval(task: str, thread_id: str, human_approves: bool) -> str:
    config = {"configurable": {"thread_id": thread_id}}

    # --- Invocation 1: runs until interrupt() ---
    print(f"Task: {task}\n")
    state_snapshot = app.invoke(
        {"task": task, "proposed_action": "", "approved": None, "result": ""},
        config=config,
    )

    proposed = state_snapshot.get("proposed_action", "(unknown)")
    print(f"Agent proposes: {proposed}")
    print(f"Human decision: {'APPROVE' if human_approves else 'REJECT'}\n")

    # --- Invocation 2: resume with human decision ---
    final = app.invoke(Command(resume=human_approves), config=config)
    return final["result"]


def main():
    print("=== 07 Human-in-the-Loop Agent ===\n")

    # Scenario 1: Human approves
    result1 = run_with_approval(
        task="Send a project status update email to the engineering team.",
        thread_id="hitl-session-1",
        human_approves=True,
    )
    print(f"Result: {result1}\n")
    print("---\n")

    # Scenario 2: Human rejects
    result2 = run_with_approval(
        task="Delete all records older than 90 days from the database.",
        thread_id="hitl-session-2",
        human_approves=False,
    )
    print(f"Result: {result2}")


if __name__ == "__main__":
    main()
