"""
04 - Planning Agent (Plan-and-Execute, LangChain)
Pattern: A planner chain produces steps; an executor chain runs each one;
         a replanner chain decides to finish or revise. A Python loop wires them.
Use cases: Complex research tasks, multi-step workflows, automated project
           breakdowns where the full plan is useful up front.
LangGraph equivalent: ../04_planning_agent.py (planner → executor ⇄ replanner graph)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    steps: List[str] = Field(description="Ordered list of steps to accomplish the objective")


class ReplanDecision(BaseModel):
    is_complete: bool = Field(description="True if the objective is fully accomplished")
    final_answer: Optional[str] = Field(default=None, description="Summary answer if complete")
    revised_steps: Optional[List[str]] = Field(default=None, description="Revised remaining steps if not complete")

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

_planner = ChatPromptTemplate.from_messages([
    ("system", "You are a task planner. Break the objective into 3-5 concrete, "
               "actionable steps. Each step should be a single clear action."),
    ("human", "Objective: {objective}"),
]) | _llm.with_structured_output(Plan)

_executor = ChatPromptTemplate.from_messages([
    ("system", "Execute this step thoroughly. Provide a clear result."),
    ("human", "Objective: {objective}\nCurrent step ({index}/{total}): {step}\nPrevious results:\n{previous}"),
]) | _llm | StrOutputParser()

_replanner = ChatPromptTemplate.from_messages([
    ("system", "Evaluate progress toward the objective. "
               "If all necessary information is gathered, produce a final answer. "
               "Otherwise, revise or continue with remaining steps.{budget_note}"),
    ("human", "Objective: {objective}\nCompleted steps:\n{completed}"),
]) | _llm.with_structured_output(ReplanDecision)

# ---------------------------------------------------------------------------
# Plan-and-execute loop
# ---------------------------------------------------------------------------

MAX_STEPS = 10  # safety cap across all replans


def run(objective: str) -> dict:
    plan = _planner.invoke({"objective": objective}).steps
    completed: List[str] = []
    final_answer: Optional[str] = None

    while plan:
        for i, step in enumerate(plan):
            result = _executor.invoke({
                "objective": objective,
                "index": i + 1,
                "total": len(plan),
                "step": step,
                "previous": "\n".join(f"  - {s}" for s in completed) or "  None yet.",
            })
            completed.append(f"Step {len(completed) + 1}: {result[:200]}")

        out_of_budget = len(completed) >= MAX_STEPS
        decision = _replanner.invoke({
            "objective": objective,
            "completed": "\n".join(f"  {s}" for s in completed),
            "budget_note": (" The step budget is used up: set is_complete to true and give the "
                            "best final answer from the results so far." if out_of_budget else ""),
        })
        if decision.is_complete or out_of_budget:
            final_answer = decision.final_answer or "\n".join(completed)
            break
        plan = decision.revised_steps or []

    if final_answer is None:  # replanner returned no steps — finish with what we have
        final_answer = "\n".join(completed)
    return {"plan": plan, "completed_steps": completed, "final_answer": final_answer}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 04 Planning Agent (Plan-and-Execute, LangChain) ===\n")
    objective = "Research the main benefits of LangGraph for AI agent development and produce a 3-point summary."
    print(f"Objective: {objective}\n")

    result = run(objective)

    print(f"Plan executed in {len(result['completed_steps'])} steps.")
    print(f"\nFinal Answer:\n{result['final_answer']}")


if __name__ == "__main__":
    main()
