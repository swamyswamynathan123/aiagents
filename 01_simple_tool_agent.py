"""
01 - Simple Tool Agent
Pattern: Manual StateGraph with a tool-call loop (no prebuilt helpers).
Use cases: Learning LangGraph mechanics, building custom tool loops,
           any agent where you need full control over tool execution.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Tavily web search — requires TAVILY_API_KEY in .env
_tavily = TavilySearch(max_results=3, include_answer=True)


def _format_results(response) -> str:
    """Flatten a Tavily response into plain text the LLM can read."""
    if not isinstance(response, dict):
        return str(response)  # handled tool error, e.g. no results
    if "error" in response:
        return f"Search failed: {response['error']}"
    lines = [f"Answer: {response['answer']}"] if response.get("answer") else []
    for r in response.get("results", []):
        lines.append(f"- {r['title']} ({r['url']}): {r['content'][:300]}")
    return "\n".join(lines) or "No results found."


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Restricting to the past day steers results toward today's conditions, not climate averages
    return _format_results(_tavily.invoke({"query": f"{city} weather today temperature", "time_range": "day"}))


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression (e.g. '42 * 7')."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is allowed"
    try:
        return str(eval(expression))  # noqa: S307 — intentionally restricted
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [get_weather, calculate]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)


def call_llm(state: AgentState) -> dict:
    response = _llm.invoke(state["messages"])
    return {"messages": [response]}


_tool_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)
_builder.add_node("llm", call_llm)
_builder.add_node("tools", _tool_node)
_builder.set_entry_point("llm")
_builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
_builder.add_edge("tools", "llm")
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 01 Simple Tool Agent ===\n")
    result = app.invoke({
        "messages": [HumanMessage(content="What's the weather in Paris and London? Also what is 42 * 7?")]
    })
    for msg in result["messages"]:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if msg.content else f"[tool_calls: {getattr(msg, 'tool_calls', [])}]"
        print(f"[{role}] {content}")


if __name__ == "__main__":
    main()
