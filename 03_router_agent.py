"""
03 - Router Agent
Pattern: Conditional edges route the query to a specialized sub-node.
Use cases: Intent classification, multi-department chatbots, query routing
           to specialist models, domain-specific handling.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RouterState(TypedDict):
    query: str
    route: str
    response: str


# ---------------------------------------------------------------------------
# Routing schema
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    route: Literal["math", "writing", "general"]
    reason: str


_router_llm = _llm.with_structured_output(RouteDecision)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: RouterState) -> dict:
    decision: RouteDecision = _router_llm.invoke([
        SystemMessage(content=(
            "Classify the user query into exactly one category:\n"
            "- math: arithmetic, algebra, statistics, numbers, calculations\n"
            "- writing: essays, poems, summaries, creative writing, editing\n"
            "- general: anything else (history, science, trivia, advice)"
        )),
        HumanMessage(content=state["query"]),
    ])
    return {"route": decision.route}


def math_node(state: RouterState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content="You are a precise math tutor. Show your work step by step."),
        HumanMessage(content=state["query"]),
    ])
    return {"response": resp.content}


def writing_node(state: RouterState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content="You are a creative writing expert. Be vivid and engaging."),
        HumanMessage(content=state["query"]),
    ])
    return {"response": resp.content}


def general_node(state: RouterState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content="You are a knowledgeable general assistant. Be concise and accurate."),
        HumanMessage(content=state["query"]),
    ])
    return {"response": resp.content}


def route_selector(state: RouterState) -> Literal["math", "writing", "general"]:
    return state["route"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(RouterState)
_builder.add_node("router", router_node)
_builder.add_node("math", math_node)
_builder.add_node("writing", writing_node)
_builder.add_node("general", general_node)
_builder.set_entry_point("router")
_builder.add_conditional_edges(
    "router",
    route_selector,
    {"math": "math", "writing": "writing", "general": "general"},
)
_builder.add_edge("math", END)
_builder.add_edge("writing", END)
_builder.add_edge("general", END)
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 03 Router Agent ===\n")
    test_queries = [
        "What is the square root of 144 plus 2 to the power of 8?",
        "Write a haiku about the Python programming language.",
        "Who invented the World Wide Web and when?",
    ]

    for query in test_queries:
        result = app.invoke({"query": query, "route": "", "response": ""})
        print(f"Query : {query}")
        print(f"Route : {result['route']}")
        print(f"Answer: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
