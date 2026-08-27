"""
09 - Handoff Agent (Swarm Pattern)
Pattern: Agents dynamically transfer control to each other based on query
         specialization. No central supervisor — agents route themselves
         by returning Command(goto=<agent_name>) objects.
Use cases: Customer service routing, tiered support systems, specialist
           agent networks where domain expertise determines routing.

Implementation note: This module implements the handoff pattern directly
using LangGraph's Command type, without requiring the separate
langgraph-swarm package. Each agent node returns a Command that routes
to the next agent or ends the conversation.

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import re
from typing import Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SwarmState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    active_agent: str
    handoff_count: int


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

AGENT_PROMPTS = {
    "triage": (
        "You are a customer service triage agent. Analyze the customer's issue and:\n"
        "1. If it's a BILLING question (invoices, charges, payments, refunds) → respond with exactly: HANDOFF:billing\n"
        "2. If it's a TECHNICAL question (bugs, errors, features, setup) → respond with exactly: HANDOFF:technical\n"
        "3. If you can resolve it yourself → provide a helpful answer and respond normally.\n"
        "Always be polite and professional."
    ),
    "billing": (
        "You are a billing specialist. Handle questions about invoices, charges, refunds, "
        "and payment methods with accuracy and empathy.\n"
        "If the issue is actually technical, respond with exactly: HANDOFF:technical\n"
        "Otherwise, resolve the billing issue helpfully."
    ),
    "technical": (
        "You are a technical support specialist. Handle bugs, errors, feature questions, "
        "and setup issues with technical precision.\n"
        "If the issue is actually a billing matter, respond with exactly: HANDOFF:billing\n"
        "Otherwise, provide clear technical guidance."
    ),
}

VALID_AGENTS = set(AGENT_PROMPTS.keys())
_HANDOFF_RE = re.compile(r"(?i)^HANDOFF\s*:\s*(\w+)")


def _make_agent_node(agent_name: str):
    """Factory that creates a node function for a given agent."""
    def agent_node(state: SwarmState) -> Command:
        resp = _llm.invoke([
            SystemMessage(content=AGENT_PROMPTS[agent_name]),
            *state["messages"],
        ])

        content = resp.content.strip()

        # Check for handoff instruction (case-insensitive, tolerates whitespace)
        _match = _HANDOFF_RE.match(content)
        if _match:
            target = _match.group(1).strip().lower()
            if target in VALID_AGENTS and target != agent_name:
                handoff_msg = AIMessage(
                    content=f"[{agent_name.title()} Agent] Transferring you to our {target} specialist..."
                )
                return Command(
                    goto=target,
                    update={
                        "messages": [handoff_msg],
                        "active_agent": target,
                        "handoff_count": state["handoff_count"] + 1,
                    },
                )

        # No handoff — respond normally and end
        final_msg = AIMessage(content=f"[{agent_name.title()} Agent] {content}")
        return Command(
            goto=END,
            update={
                "messages": [final_msg],
                "active_agent": agent_name,
            },
        )

    agent_node.__name__ = f"{agent_name}_agent"
    return agent_node


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(SwarmState)
_builder.add_node("triage", _make_agent_node("triage"))
_builder.add_node("billing", _make_agent_node("billing"))
_builder.add_node("technical", _make_agent_node("technical"))
_builder.add_edge(START, "triage")
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_query(query: str) -> None:
    print(f"Customer: {query}")
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "active_agent": "triage",
        "handoff_count": 0,
    })
    handoffs = result["handoff_count"]
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:300]}{'...' if len(msg.content) > 300 else ''}")
    print(f"  [Handoffs: {handoffs}, Final agent: {result['active_agent']}]\n")


def main():
    print("=== 09 Handoff Agent (Swarm Pattern) ===\n")

    queries = [
        "I was charged twice on my last invoice. Can you help?",
        "I'm getting a 500 error when I try to log in to the dashboard.",
        "What are your business hours?",
    ]

    for query in queries:
        run_query(query)


if __name__ == "__main__":
    main()
