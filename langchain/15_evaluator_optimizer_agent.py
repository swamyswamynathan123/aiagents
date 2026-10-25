"""
15 - Evaluator-Optimizer Agent (LangChain)
Pattern: A creative generator chain and a deterministic evaluator chain
         (rubric + structured output) alternate in a Python loop until the
         score hits the target or iterations run out.
Use cases: Marketing copy, prompt optimization, code quality loops, any
           output with a measurable rubric.
LangGraph equivalent: ../15_evaluator_optimizer_agent.py (generator ⇄ evaluator graph)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

_generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)  # creative
_evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)    # deterministic judge

# ---------------------------------------------------------------------------
# Evaluator schema + rubric
# ---------------------------------------------------------------------------

class EvalResult(BaseModel):
    score: int = Field(description="Quality score from 1 (very poor) to 10 (excellent)", ge=1, le=10)
    passed: bool = Field(description="True if score meets or exceeds the target threshold")
    strengths: str = Field(description="What the output does well")
    weaknesses: str = Field(description="Specific issues that reduce the score")
    feedback: str = Field(description="Concrete, actionable improvement instructions")


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
# Chains
# ---------------------------------------------------------------------------

_generator = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class marketing copywriter."),
    ("human", "{prompt}"),
]) | _generator_llm | StrOutputParser()

_evaluator = ChatPromptTemplate.from_messages([
    ("system", "You are a rigorous marketing copy evaluator.\n\n{rubric}"),
    ("human", "Task given: {task}\n\nProduct description to evaluate:\n{output}"),
]) | _evaluator_llm.with_structured_output(EvalResult)

# ---------------------------------------------------------------------------
# Optimize loop
# ---------------------------------------------------------------------------

def run(task: str, target_score: int = 8, max_iterations: int = 4) -> dict:
    output, score, feedback, weaknesses, passed = "", 0, "", "", False
    score_history = []
    iteration = 0

    while iteration < max_iterations:
        if iteration == 0:
            prompt = f"Task: {task}\n\nWrite a compelling product description. Be creative and persuasive."
        else:
            prompt = (
                f"Task: {task}\n\n"
                f"Previous version (score {score}/10):\n{output}\n\n"
                f"Evaluator feedback:\n{feedback}\n\n"
                f"Weaknesses to fix: {weaknesses or 'See feedback above'}\n\n"
                "Rewrite significantly improved version addressing all feedback:"
            )
        output = _generator.invoke({"prompt": prompt})
        iteration += 1

        result = _evaluator.invoke({"rubric": RUBRIC.format(target=target_score), "task": task, "output": output})
        score, passed, feedback, weaknesses = result.score, result.passed, result.feedback, result.weaknesses
        score_history.append(score)
        if passed:
            break

    return {
        "output": output, "score": score, "feedback": feedback, "weaknesses": weaknesses,
        "passed": passed, "iteration": iteration, "score_history": score_history,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 15 Evaluator-Optimizer Agent (LangChain) ===\n")
    task = "Write a product description for a smart water bottle that tracks hydration and syncs with a fitness app."
    target = 8
    max_iter = 4
    print(f"Task: {task}")
    print(f"Target score: {target}/10 | Max iterations: {max_iter}\n")

    result = run(task, target_score=target, max_iterations=max_iter)

    print(f"Score progression: {result['score_history']}")
    print(f"Final score: {result['score']}/10 | Passed: {result['passed']}")
    print(f"Iterations: {result['iteration']}")
    print(f"\nFinal Output:\n{result['output']}")


if __name__ == "__main__":
    main()
