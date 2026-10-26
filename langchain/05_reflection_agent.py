"""
05 - Reflection Agent (LangChain)
Pattern: A generator chain writes a draft; a critic chain (structured output)
         scores it; a Python loop revises until approved or max iterations.
Use cases: Content quality improvement, code review loops, essay refinement,
           any task where self-critique raises output quality.
LangGraph equivalent: ../05_reflection_agent.py (generator ⇄ critic cycle)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0.7)
_critic_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Critic schema
# ---------------------------------------------------------------------------

class CritiqueResult(BaseModel):
    approved: bool = Field(description="True if the draft meets quality standards")
    score: int = Field(description="Quality score from 1 (poor) to 10 (excellent)")
    feedback: str = Field(description="Specific, actionable feedback for improvement")

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

_generator = ChatPromptTemplate.from_messages([
    ("system", "You are an expert writer. Produce clear, compelling, high-quality content."),
    ("human", "{prompt}"),
]) | _llm | StrOutputParser()

_critic = ChatPromptTemplate.from_messages([
    ("system", "You are a rigorous editor. Evaluate the draft on clarity, accuracy, "
               "and impact. Be specific and constructive. Approve only if score >= 8."),
    ("human", "Task: {task}\n\nDraft to evaluate:\n{draft}"),
]) | _critic_llm.with_structured_output(CritiqueResult)

# ---------------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------------

def run(task: str, max_iterations: int = 3) -> dict:
    draft, critique, score, approved, iteration = "", "", 0, False, 0

    while iteration < max_iterations:
        if iteration == 0:
            prompt = f"Task: {task}\n\nWrite a high-quality response:"
        else:
            prompt = (
                f"Task: {task}\n\n"
                f"Previous draft:\n{draft}\n\n"
                f"Critic feedback (score {score}/10):\n{critique}\n\n"
                "Revise the draft to address the feedback. Improve quality significantly:"
            )
        draft = _generator.invoke({"prompt": prompt})
        iteration += 1

        result = _critic.invoke({"task": task, "draft": draft})
        critique, score, approved = result.feedback, result.score, result.approved
        if approved:
            break

    return {"draft": draft, "critique": critique, "score": score, "iteration": iteration, "approved": approved}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 05 Reflection Agent (LangChain) ===\n")
    task = "Write a compelling 3-sentence product pitch for an AI-powered coding assistant."
    print(f"Task: {task}\n")

    result = run(task, max_iterations=3)

    print(f"Completed in {result['iteration']} iteration(s).")
    print(f"Approved: {result['approved']}")
    print(f"\nFinal Draft:\n{result['draft']}")


if __name__ == "__main__":
    main()
