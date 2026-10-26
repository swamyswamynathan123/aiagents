"""
13 - Parallel Agent (Fan-Out via Send)
Pattern: Send() from langgraph.types dispatches multiple independent tasks
         in parallel. An operator.add reducer aggregates all results into
         a single list (map-reduce).
Use cases: Batch document analysis, parallel web scraping, concurrent
           evaluations, any workflow where items can be processed independently.

How Send() works:
  - A fan-out function returns List[Send(node_name, state_dict)]
  - LangGraph dispatches all Sends concurrently
  - Each parallel node's return is merged into the parent state via reducer
  - operator.add on List[str] appends each parallel result

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
import operator
from typing import Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0.3)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class ItemState(TypedDict):
    """State for a single parallel worker processing one item."""
    item: str
    category: str  # what dimension to analyze


class AggregateState(TypedDict):
    """Parent state that collects all parallel results."""
    items: List[str]
    category: str
    analyses: Annotated[List[str], operator.add]  # reducer: appends across parallel workers
    summary: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def analyze_item(state: ItemState) -> dict:
    """Process a single item. Called in parallel for each item via Send()."""
    resp = _llm.invoke([
        SystemMessage(content=(
            f"You are an expert analyst evaluating programming languages. "
            f"Category to analyze: {state['category']}. "
            f"Be concise — 2-3 sentences max."
        )),
        HumanMessage(content=f"Analyze '{state['item']}' for {state['category']}."),
    ])
    # Return writes into AggregateState.analyses via operator.add
    return {"analyses": [f"[{state['item']}] {resp.content}"]}


def aggregate_results(state: AggregateState) -> dict:
    """Combine all parallel analyses into a final summary."""
    all_analyses = "\n\n".join(state["analyses"])
    resp = _llm.invoke([
        SystemMessage(content=(
            "You are a synthesizer. Combine the individual analyses into a "
            "clear, comparative summary. Highlight key differences."
        )),
        HumanMessage(content=(
            f"Category: {state['category']}\n\n"
            f"Individual analyses:\n{all_analyses}\n\n"
            "Write a concise comparative summary:"
        )),
    ])
    return {"summary": resp.content}


# ---------------------------------------------------------------------------
# Fan-out function
# ---------------------------------------------------------------------------

def fan_out(state: AggregateState) -> List[Send]:
    """Returns one Send() per item — triggers parallel execution."""
    return [
        Send("analyze_item", {"item": item, "category": state["category"]})
        for item in state["items"]
    ]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

from langgraph.graph import START  # noqa: E402

_builder = StateGraph(AggregateState)
_builder.add_node("analyze_item", analyze_item)
_builder.add_node("aggregate", aggregate_results)

# Fan-out: START → parallel analyze_item nodes
_builder.add_conditional_edges(START, fan_out)

# Fan-in: all analyze_item nodes → aggregate (LangGraph waits for all)
_builder.add_edge("analyze_item", "aggregate")
_builder.add_edge("aggregate", END)
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 13 Parallel Agent (Fan-Out Map-Reduce) ===\n")

    items = ["Python", "Rust", "TypeScript", "Go"]
    category = "ecosystem maturity and tooling"

    print(f"Analyzing {len(items)} items in parallel: {items}")
    print(f"Category: {category}\n")

    result = app.invoke({
        "items": items,
        "category": category,
        "analyses": [],
        "summary": "",
    })

    print(f"Parallel analyses collected: {len(result['analyses'])}")
    print("\nIndividual Results:")
    for analysis in result["analyses"]:
        print(f"  {analysis[:150]}{'...' if len(analysis) > 150 else ''}")

    print(f"\nAggregated Summary:\n{result['summary']}")


if __name__ == "__main__":
    main()
