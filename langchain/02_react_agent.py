"""
02 - ReAct Agent (LangChain)
Pattern: Reasoning + Acting loop built from an LCEL chain
         (ChatPromptTemplate | llm.bind_tools) driven by a Python loop.
Use cases: General-purpose tool-using assistants, research bots, Q&A with lookups.

NOTE: In LangChain 1.x, `create_agent` is built on LangGraph and the legacy
`AgentExecutor` lives in `langchain-classic`. This module keeps to
langchain-core primitives so it runs without LangGraph.

LangGraph equivalent: ../02_react_agent.py (prebuilt create_react_agent)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    return _format_results(_tavily.invoke({"query": query}))


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    return _format_results(
        _tavily.invoke({"query": f"{ticker.upper()} stock price today", "topic": "finance"})
    )


@tool
def get_news_headlines(topic: str) -> str:
    """Get recent news headlines for a topic."""
    return _format_results(
        _tavily.invoke({"query": f"{topic} latest news", "topic": "news", "time_range": "week"})
    )


TOOLS = [search_web, get_stock_price, get_news_headlines]
TOOL_MAP = {t.name: t for t in TOOLS}

# ---------------------------------------------------------------------------
# Agent — LCEL chain: prompt → tool-bound LLM
# ---------------------------------------------------------------------------

_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Think step by step and use tools "
               "whenever you need current information. Cite what the tools returned."),
    MessagesPlaceholder("messages"),
])
_agent = _prompt | ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)

MAX_STEPS = 10


def run(query: str) -> dict:
    messages = [HumanMessage(content=query)]
    for _ in range(MAX_STEPS):
        response = _agent.invoke({"messages": messages})  # Reason
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:  # Act, then Observe on the next pass
            output = TOOL_MAP[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 02 ReAct Agent (LangChain) ===\n")

    queries = [
        "What's the stock price of AAPL and NVDA?",
        "Search for recent news about LangGraph and summarize what you find.",
    ]

    for query in queries:
        print(f"Query: {query}")
        final = run(query)["messages"][-1].content
        print(f"Answer: {final[:300]}{'...' if len(final) > 300 else ''}\n")


if __name__ == "__main__":
    main()
