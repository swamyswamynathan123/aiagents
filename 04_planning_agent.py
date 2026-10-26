"""
04 - Planning Agent (Plan-and-Execute)
Pattern: Planner generates a multi-step plan; Executor runs each step;
         Re-planner adjusts or finalizes based on results.
Use cases: Complex research tasks, multi-step workflows, automated project
           management, tasks that benefit from upfront decomposition.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanState(TypedDict):
    objective: str
    plan: List[str]
    current_step: int
    completed_steps: List[str]
    final_answer: Optional[str]


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    steps: List[str] = Field(description="Ordered list of steps to accomplish the objective")


class ReplanDecision(BaseModel):
    is_complete: bool = Field(description="True if the objective is fully accomplished")
    final_answer: Optional[str] = Field(default=None, description="Summary answer if complete")
    revised_steps: Optional[List[str]] = Field(default=None, description="Revised remaining steps if not complete")


_planner_llm = _llm.with_structured_output(Plan)
_replanner_llm = _llm.with_structured_output(ReplanDecision)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: PlanState) -> dict:
    plan: Plan = _planner_llm.invoke([
        SystemMessage(content=(
            "You are a task planner. Break the objective into 3-5 concrete, "
            "actionable steps. Each step should be a single clear action."
        )),
        HumanMessage(content=f"Objective: {state['objective']}"),
    ])
    return {"plan": plan.steps, "current_step": 0, "completed_steps": []}


def executor_node(state: PlanState) -> dict:
    step = state["plan"][state["current_step"]]
    resp = _llm.invoke([
        SystemMessage(content="Execute this step thoroughly. Provide a clear result."),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n"
            f"Current step ({state['current_step'] + 1}/{len(state['plan'])}): {step}\n"
            "Previous results:\n"
            + ("\n".join(f"  - {s}" for s in state["completed_steps"]) if state["completed_steps"] else "  None yet.")
        )),
    ])
    completed = list(state["completed_steps"]) + [f"Step {state['current_step'] + 1}: {resp.content[:200]}"]
    return {"completed_steps": completed, "current_step": state["current_step"] + 1}


# LangGraph 1.2's default recursion_limit is ~10k, so it won't stop a replanner
# that keeps asking for more steps — cap executed steps explicitly.
MAX_STEPS = 10


def replanner_node(state: PlanState) -> dict:
    out_of_budget = len(state["completed_steps"]) >= MAX_STEPS
    decision: ReplanDecision = _replanner_llm.invoke([
        SystemMessage(content=(
            "Evaluate progress toward the objective. "
            "If all necessary information is gathered, produce a final answer. "
            "Otherwise, revise or continue with remaining steps."
            + (" The step budget is used up: set is_complete to true and give the "
               "best final answer from the results so far." if out_of_budget else "")
        )),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n"
            f"Completed steps:\n" + "\n".join(
                f"  {s}" for s in state["completed_steps"]
            )
        )),
    ])
    if decision.is_complete or out_of_budget:
        return {"final_answer": decision.final_answer or "\n".join(state["completed_steps"])}
    new_plan = decision.revised_steps or state["plan"][state["current_step"]:]
    if not new_plan:  # nothing left to execute — finish with what we have
        return {"final_answer": decision.final_answer or "\n".join(state["completed_steps"])}
    return {"plan": new_plan, "current_step": 0}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def after_executor(state: PlanState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "replanner"
    return "executor"


def after_replanner(state: PlanState) -> str:
    if state.get("final_answer"):
        return END
    return "executor"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(PlanState)
_builder.add_node("planner", planner_node)
_builder.add_node("executor", executor_node)
_builder.add_node("replanner", replanner_node)
_builder.set_entry_point("planner")
_builder.add_edge("planner", "executor")
_builder.add_conditional_edges("executor", after_executor, {"replanner": "replanner", "executor": "executor"})
_builder.add_conditional_edges("replanner", after_replanner, {END: END, "executor": "executor"})
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 04 Planning Agent (Plan-and-Execute) ===\n")
    objective = "Research the main benefits of LangGraph for AI agent development and produce a 3-point summary."
    print(f"Objective: {objective}\n")

    result = app.invoke({
        "objective": objective,
        "plan": [],
        "current_step": 0,
        "completed_steps": [],
        "final_answer": None,
    })

    print(f"Plan executed in {len(result['completed_steps'])} steps.")
    print(f"\nFinal Answer:\n{result['final_answer']}")


if __name__ == "__main__":
    main()
