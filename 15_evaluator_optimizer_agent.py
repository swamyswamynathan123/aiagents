"""
15 - Evaluator-Optimizer Agent
Pattern: A Generator produces output; a separate Evaluator LLM scores it
         against an explicit rubric; the loop continues until the target
         score is reached or max_iterations is exhausted.
         Unlike Reflection (self-critique), the Evaluator is a distinct
         judge with a structured scoring rubric and pass/fail threshold.
Use cases: Marketing copy optimization, legal/compliance document review,
           API documentation quality gates, automated A/B testing of prompts,
           any output with measurable quality criteria.
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

_generator_llm = ChatOpenAI(model=MODEL, temperature=0.8)  # creative
_evaluator_llm = ChatOpenAI(model=MODEL, temperature=0)    # deterministic judge

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EvalOptState(TypedDict):
    task: str
    output: str
    score: int
    feedback: str
    iteration: int
    max_iterations: int
    target_score: int
    passed: bool
    weaknesses: str
    score_history: list


# ---------------------------------------------------------------------------
# Evaluator schema
# ---------------------------------------------------------------------------

class EvalResult(BaseModel):
    score: int = Field(description="Quality score from 1 (very poor) to 10 (excellent)", ge=1, le=10)
    passed: bool = Field(description="True if score meets or exceeds the target threshold")
    strengths: str = Field(description="What the output does well")
    weaknesses: str = Field(description="Specific issues that reduce the score")
    feedback: str = Field(description="Concrete, actionable improvement instructions")


_structured_evaluator = _evaluator_llm.with_structured_output(EvalResult)

# ---------------------------------------------------------------------------
# Rubric (modify per use case)
# ---------------------------------------------------------------------------

RUBRIC = """
Evaluate the product description on these dimensions (each 0-2 points):
1. Clarity (0-2): Is it immediately understandable to a non-expert?
2. Benefit focus (0-2): Does it emphasize customer benefits over features?
3. Specificity (0-2): Does it include concrete, credible details?
4. Emotional appeal (0-2): Does it create desire or excitement?
5. Call to action (0-2): Does it end with a clear next step?

Total: 10 points. Pass threshold: {target}/10.
"""

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generator_node(state: EvalOptState) -> dict:
    if state["iteration"] == 0:
        prompt = (
            f"Task: {state['task']}\n\n"
            "Write a compelling product description. Be creative and persuasive."
        )
    else:
        prompt = (
            f"Task: {state['task']}\n\n"
            f"Previous version (score {state['score']}/10):\n{state['output']}\n\n"
            f"Evaluator feedback:\n{state['feedback']}\n\n"
            f"Weaknesses to fix: {state.get('weaknesses', 'See feedback above')}\n\n"
            "Rewrite significantly improved version addressing all feedback:"
        )

    resp = _generator_llm.invoke([
        SystemMessage(content="You are a world-class marketing copywriter."),
        HumanMessage(content=prompt),
    ])
    return {
        "output": resp.content,
        "iteration": state["iteration"] + 1,
    }


def evaluator_node(state: EvalOptState) -> dict:
    rubric = RUBRIC.format(target=state["target_score"])
    result: EvalResult = _structured_evaluator.invoke([
        SystemMessage(content=f"You are a rigorous marketing copy evaluator.\n\n{rubric}"),
        HumanMessage(content=f"Task given: {state['task']}\n\nProduct description to evaluate:\n{state['output']}"),
    ])

    history = list(state.get("score_history", [])) + [result.score]

    return {
        "score": result.score,
        "passed": result.passed,
        "feedback": result.feedback,
        "weaknesses": result.weaknesses,
        "score_history": history,
    }


def eval_router(state: EvalOptState) -> str:
    if state["passed"] or state["iteration"] >= state["max_iterations"]:
        return END
    return "generator"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(EvalOptState)
_builder.add_node("generator", generator_node)
_builder.add_node("evaluator", evaluator_node)
_builder.set_entry_point("generator")
_builder.add_edge("generator", "evaluator")
_builder.add_conditional_edges("evaluator", eval_router, {"generator": "generator", END: END})
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 15 Evaluator-Optimizer Agent ===\n")
    task = "Write a product description for a smart water bottle that tracks hydration and syncs with a fitness app."
    target = 8
    max_iter = 4

    print(f"Task: {task}")
    print(f"Target score: {target}/10 | Max iterations: {max_iter}\n")

    result = app.invoke({
        "task": task,
        "output": "",
        "score": 0,
        "feedback": "",
        "weaknesses": "",
        "iteration": 0,
        "max_iterations": max_iter,
        "target_score": target,
        "passed": False,
        "score_history": [],
    })

    print(f"Score progression: {result['score_history']}")
    print(f"Final score: {result['score']}/10 | Passed: {result['passed']}")
    print(f"Iterations: {result['iteration']}")
    print(f"\nFinal Output:\n{result['output']}")


if __name__ == "__main__":
    main()
