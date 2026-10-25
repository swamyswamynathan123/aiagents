"""
02 - ReAct Agent
Pattern: Reasoning + Acting loop using the langgraph.prebuilt helper.
Use cases: General-purpose tool-using assistants, research bots, Q&A with lookups.

DEPRECATION NOTE: `create_react_agent` from `langgraph.prebuilt` is deprecated
in LangGraph 1.2.x. The successor is `create_agent` from `langchain` (langchain>=0.3).
This module is kept as a reference implementation. For new projects prefer the
langchain package's `create_agent`.

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent  # deprecated in 1.2.x

load_dotenv()

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

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
# create_react_agent returns a compiled StateGraph directly — no manual wiring needed.

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
app = create_react_agent(_llm, tools=TOOLS)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 02 ReAct Agent ===\n")

    queries = [
        "What's the stock price of AAPL and NVDA?",
        "Search for recent news about LangGraph and summarize what you find.",
    ]

    for query in queries:
        print(f"Query: {query}")
        result = app.invoke({"messages": [HumanMessage(content=query)]})
        final = result["messages"][-1].content
        print(f"Answer: {final[:300]}{'...' if len(final) > 300 else ''}\n")


if __name__ == "__main__":
    main()
