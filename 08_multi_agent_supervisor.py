"""
08 - Multi-Agent Supervisor
Pattern: A Supervisor node decides which worker to delegate to next.
         Workers (Researcher, Writer) complete their task and return to
         the Supervisor, which loops until the objective is met.
Use cases: Content pipelines (research then write), complex analyses with
           multiple expert roles, automated report generation, any workflow
           requiring sequential specialist agents.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import Annotated, List, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    objective: str
    research_result: str
    written_result: str
    next_worker: str
    iteration: int


# ---------------------------------------------------------------------------
# Routing schema
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="Which worker to call next, or FINISH if the objective is complete"
    )
    reason: str = Field(description="One-sentence rationale for this routing decision")


_supervisor_llm = _llm.with_structured_output(RoutingDecision)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def supervisor_node(state: SupervisorState) -> dict:
    status = []
    if state["research_result"]:
        status.append(f"Research complete: {state['research_result'][:100]}...")
    if state["written_result"]:
        status.append(f"Writing complete: {state['written_result'][:100]}...")

    decision: RoutingDecision = _supervisor_llm.invoke([
        SystemMessage(content=(
            "You are a project supervisor coordinating a research-and-writing team.\n"
            "Workers: 'researcher' gathers information, 'writer' drafts content.\n"
            "Routing rules:\n"
            "  1. Start with researcher if no research exists.\n"
            "  2. Send to writer when research is complete but writing is not.\n"
            "  3. FINISH when both research and writing are done.\n"
            "Never route to a worker that has already completed its work."
        )),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n"
            f"Current status:\n" + ("\n".join(f"  - {s}" for s in status) if status else "  Nothing done yet.")
        )),
    ])
    return {"next_worker": decision.next}


def researcher_node(state: SupervisorState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content=(
            "You are a research specialist. Gather key facts, data points, and "
            "insights relevant to the objective. Be thorough and structured."
        )),
        HumanMessage(content=f"Research this objective: {state['objective']}"),
    ])
    return {
        "research_result": resp.content,
        "messages": [HumanMessage(content=f"[Researcher]: {resp.content[:150]}...")],
        "iteration": state["iteration"] + 1,
    }


def writer_node(state: SupervisorState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content=(
            "You are an expert writer. Use the provided research to write "
            "clear, engaging, well-structured content."
        )),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n\n"
            f"Research findings:\n{state['research_result']}\n\n"
            "Write the final content now:"
        )),
    ])
    return {
        "written_result": resp.content,
        "messages": [HumanMessage(content=f"[Writer]: {resp.content[:150]}...")],
        "iteration": state["iteration"] + 1,
    }


def supervisor_router(state: SupervisorState) -> str:
    next_w = state["next_worker"]
    if next_w == "FINISH" or state["iteration"] >= 6:
        return END
    return next_w


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(SupervisorState)
_builder.add_node("supervisor", supervisor_node)
_builder.add_node("researcher", researcher_node)
_builder.add_node("writer", writer_node)
_builder.set_entry_point("supervisor")
_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {"researcher": "researcher", "writer": "writer", END: END},
)
_builder.add_edge("researcher", "supervisor")
_builder.add_edge("writer", "supervisor")
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 08 Multi-Agent Supervisor ===\n")
    objective = "Research LangGraph's key features and write a concise 2-paragraph overview."
    print(f"Objective: {objective}\n")

    result = app.invoke({
        "messages": [HumanMessage(content=objective)],
        "objective": objective,
        "research_result": "",
        "written_result": "",
        "next_worker": "",
        "iteration": 0,
    })

    print(f"Completed in {result['iteration']} worker invocation(s).\n")
    print("Research Summary:")
    print(result["research_result"][:400] + ("..." if len(result["research_result"]) > 400 else ""))
    print("\nFinal Written Output:")
    print(result["written_result"])


if __name__ == "__main__":
    main()
