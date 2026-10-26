"""
05 - Reflection Agent
Pattern: Generator produces output; Critic evaluates it; loop repeats
         until quality threshold is met or max_iterations is reached.
Use cases: Self-improving content generation, iterative code writing,
           essay drafting, any task where quality matters more than speed.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0.7)
_critic_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    score: int
    iteration: int
    max_iterations: int
    approved: bool


# ---------------------------------------------------------------------------
# Critic schema
# ---------------------------------------------------------------------------

class CritiqueResult(BaseModel):
    approved: bool = Field(description="True if the draft meets quality standards")
    score: int = Field(description="Quality score from 1 (poor) to 10 (excellent)")
    feedback: str = Field(description="Specific, actionable feedback for improvement")


_structured_critic = _critic_llm.with_structured_output(CritiqueResult)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generator_node(state: ReflectionState) -> dict:
    if state["iteration"] == 0:
        prompt = f"Task: {state['task']}\n\nWrite a high-quality response:"
    else:
        prompt = (
            f"Task: {state['task']}\n\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Critic feedback (score {state.get('score', '?')}/10):\n{state['critique']}\n\n"
            "Revise the draft to address the feedback. Improve quality significantly:"
        )

    resp = _llm.invoke([
        SystemMessage(content="You are an expert writer. Produce clear, compelling, high-quality content."),
        HumanMessage(content=prompt),
    ])
    return {"draft": resp.content, "iteration": state["iteration"] + 1}


def critic_node(state: ReflectionState) -> dict:
    result: CritiqueResult = _structured_critic.invoke([
        SystemMessage(content=(
            "You are a rigorous editor. Evaluate the draft on clarity, accuracy, "
            "and impact. Be specific and constructive. Approve only if score >= 8."
        )),
        HumanMessage(content=f"Task: {state['task']}\n\nDraft to evaluate:\n{state['draft']}"),
    ])
    return {"critique": result.feedback, "approved": result.approved, "score": result.score}


def reflection_router(state: ReflectionState) -> str:
    if state["approved"] or state["iteration"] >= state["max_iterations"]:
        return END
    return "generator"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(ReflectionState)
_builder.add_node("generator", generator_node)
_builder.add_node("critic", critic_node)
_builder.set_entry_point("generator")
_builder.add_edge("generator", "critic")
_builder.add_conditional_edges("critic", reflection_router, {"generator": "generator", END: END})
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 05 Reflection Agent ===\n")
    task = "Write a compelling 3-sentence product pitch for an AI-powered coding assistant."
    print(f"Task: {task}\n")

    result = app.invoke({
        "task": task,
        "draft": "",
        "critique": "",
        "score": 0,
        "iteration": 0,
        "max_iterations": 3,
        "approved": False,
    })

    print(f"Completed in {result['iteration']} iteration(s).")
    print(f"Approved: {result['approved']}")
    print(f"\nFinal Draft:\n{result['draft']}")


if __name__ == "__main__":
    main()
