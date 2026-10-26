"""
01 - Simple Tool Agent (LangChain)
Pattern: Hand-written tool-call loop using llm.bind_tools() and a plain
         Python while-loop — no graph, no prebuilt executor.
Use cases: Learning how tool calling works under the hood, building custom
           tool loops, any agent where you need full control over execution.
LangGraph equivalent: ../01_simple_tool_agent.py (StateGraph + ToolNode + conditional edge)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

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
TOOL_MAP = {t.name: t for t in TOOLS}

_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# Agent loop — replaces the graph's llm → tools → llm cycle
# ---------------------------------------------------------------------------

MAX_STEPS = 10


def run(query: str) -> dict:
    messages: List[BaseMessage] = [HumanMessage(content=query)]
    for _ in range(MAX_STEPS):
        response = _llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            output = TOOL_MAP[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 01 Simple Tool Agent (LangChain) ===\n")
    result = run("What's the weather in Paris and London? Also what is 42 * 7?")
    for msg in result["messages"]:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if msg.content else f"[tool_calls: {getattr(msg, 'tool_calls', [])}]"
        print(f"[{role}] {content}")


if __name__ == "__main__":
    main()
