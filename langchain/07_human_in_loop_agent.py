"""
07 - Human-in-the-Loop Agent (LangChain)
Pattern: The workflow is split into two calls — propose_action() returns a
         plan for review, resolve() executes or cancels it. The application
         holds the pending state between calls (no checkpointer needed).
Use cases: Approval workflows, high-stakes actions (payments, deletions,
           outbound emails), compliance gates, agent oversight.
LangGraph equivalent: ../07_human_in_loop_agent.py (interrupt() + Command(resume=...)),
                      where the graph itself pauses and persists mid-run.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

_planner = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that takes actions on behalf of users. "
               "Describe the specific action you would take to accomplish the task. "
               "Be concise and specific (1-2 sentences)."),
    ("human", "Task: {task}"),
]) | _llm | StrOutputParser()

_executor = ChatPromptTemplate.from_messages([
    ("system", "Execute the following action and confirm completion."),
    ("human", "Action: {action}"),
]) | _llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Two-step API — the human decision happens between these calls
# ---------------------------------------------------------------------------

def propose_action(task: str) -> str:
    """Step 1: plan the action and return it for human review."""
    return _planner.invoke({"task": task})


def resolve(proposed_action: str, approved: bool) -> str:
    """Step 2: execute if approved, otherwise cancel."""
    if not approved:
        return f"[CANCELLED] Action rejected by human: '{proposed_action}'"
    return f"[EXECUTED] {_executor.invoke({'action': proposed_action})}"


def run_with_approval(task: str, human_approves: bool) -> str:
    print(f"Task: {task}\n")
    proposed = propose_action(task)
    print(f"Agent proposes: {proposed}")
    print(f"Human decision: {'APPROVE' if human_approves else 'REJECT'}\n")
    return resolve(proposed, human_approves)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 07 Human-in-the-Loop Agent (LangChain) ===\n")

    # Scenario 1: Human approves
    result1 = run_with_approval("Send a project status update email to the engineering team.", human_approves=True)
    print(f"Result: {result1}\n")
    print("---\n")

    # Scenario 2: Human rejects
    result2 = run_with_approval("Delete all records older than 90 days from the database.", human_approves=False)
    print(f"Result: {result2}")


if __name__ == "__main__":
    main()
