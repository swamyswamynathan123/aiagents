"""
08 - Multi-Agent Supervisor (LangChain)
Pattern: A supervisor chain (structured output) picks the next worker;
         worker chains (researcher, writer) do the work; a Python loop
         dispatches until the supervisor says FINISH.
Use cases: Content pipelines (research then write), complex analyses with
           specialist roles, any task decomposable into expert sub-agents.
LangGraph equivalent: ../08_multi_agent_supervisor.py (supervisor node + conditional edges)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import Literal
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# Routing schema
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="Which worker to call next, or FINISH if the objective is complete"
    )
    reason: str = Field(description="One-sentence rationale for this routing decision")

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

_supervisor = ChatPromptTemplate.from_messages([
    ("system",
     "You are a project supervisor coordinating a research-and-writing team.\n"
     "Workers: 'researcher' gathers information, 'writer' drafts content.\n"
     "Routing rules:\n"
     "  1. Start with researcher if no research exists.\n"
     "  2. Send to writer when research is complete but writing is not.\n"
     "  3. FINISH when both research and writing are done.\n"
     "Never route to a worker that has already completed its work."),
    ("human", "Objective: {objective}\nCurrent status:\n{status}"),
]) | _llm.with_structured_output(RoutingDecision)

_researcher = ChatPromptTemplate.from_messages([
    ("system", "You are a research specialist. Gather key facts, data points, and "
               "insights relevant to the objective. Be thorough and structured."),
    ("human", "Research this objective: {objective}"),
]) | _llm | StrOutputParser()

_writer = ChatPromptTemplate.from_messages([
    ("system", "You are an expert writer. Use the provided research to write "
               "clear, engaging, well-structured content."),
    ("human", "Objective: {objective}\n\nResearch findings:\n{research}\n\nWrite the final content now:"),
]) | _llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Supervisor loop
# ---------------------------------------------------------------------------

MAX_WORKER_CALLS = 6


def run(objective: str) -> dict:
    research, written, iteration = "", "", 0

    while iteration < MAX_WORKER_CALLS:
        status = []
        if research:
            status.append(f"Research complete: {research[:100]}...")
        if written:
            status.append(f"Writing complete: {written[:100]}...")
        decision = _supervisor.invoke({
            "objective": objective,
            "status": "\n".join(f"  - {s}" for s in status) or "  Nothing done yet.",
        })

        if decision.next == "FINISH":
            break
        if decision.next == "researcher":
            research = _researcher.invoke({"objective": objective})
        else:
            written = _writer.invoke({"objective": objective, "research": research})
        iteration += 1

    return {"research_result": research, "written_result": written, "iteration": iteration}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 08 Multi-Agent Supervisor (LangChain) ===\n")
    objective = "Research LangGraph's key features and write a concise 2-paragraph overview."
    print(f"Objective: {objective}\n")

    result = run(objective)

    print(f"Completed in {result['iteration']} worker invocation(s).\n")
    print("Research Summary:")
    print(result["research_result"][:400] + ("..." if len(result["research_result"]) > 400 else ""))
    print("\nFinal Written Output:")
    print(result["written_result"])


if __name__ == "__main__":
    main()
