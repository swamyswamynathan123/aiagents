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
from langgraph.prebuilt import create_react_agent  # deprecated in 1.2.x

load_dotenv()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    return (
        f"Search results for '{query}':\n"
        f"1. '{query}' is a popular topic with recent developments in 2026.\n"
        f"2. Experts highlight key trends including AI integration and automation.\n"
        f"3. Latest studies show significant adoption across industries."
    )


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    mock_prices = {"AAPL": 195.42, "GOOGL": 178.30, "MSFT": 425.10, "NVDA": 875.60}
    price = mock_prices.get(ticker.upper(), 150.00)
    return f"{ticker.upper()}: ${price:.2f} (mock data)"


@tool
def get_news_headlines(topic: str) -> str:
    """Get recent news headlines for a topic."""
    return (
        f"Top headlines for '{topic}':\n"
        f"• Major breakthrough announced in {topic} space — TechCrunch\n"
        f"• Industry leaders discuss future of {topic} — Reuters\n"
        f"• New regulations proposed for {topic} sector — Bloomberg"
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
