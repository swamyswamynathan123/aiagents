"""
09 - Handoff Agent (Swarm Pattern, LangChain)
Pattern: Each agent is a prompt | llm chain. An agent hands off by replying
         "HANDOFF:<agent>"; a Python loop switches the active chain until one
         answers directly (or the handoff cap is reached).
Use cases: Customer service routing, specialist escalation, peer-to-peer
           multi-agent systems without a central coordinator.
LangGraph equivalent: ../09_handoff_agent.py (nodes returning Command(goto=...))
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

# One chain per agent: its system prompt + the shared conversation
AGENTS = {
    name: ChatPromptTemplate.from_messages([("system", prompt), MessagesPlaceholder("messages")]) | _llm | StrOutputParser()
    for name, prompt in AGENT_PROMPTS.items()
}

_HANDOFF_RE = re.compile(r"(?i)^HANDOFF\s*:\s*(\w+)")
MAX_HANDOFFS = 3

# ---------------------------------------------------------------------------
# Handoff loop
# ---------------------------------------------------------------------------

def run(query: str) -> dict:
    messages = [HumanMessage(content=query)]
    active, handoffs = "triage", 0

    while True:
        content = AGENTS[active].invoke({"messages": messages}).strip()
        match = _HANDOFF_RE.match(content)
        target = match.group(1).lower() if match else None

        if target in AGENTS and target != active and handoffs < MAX_HANDOFFS:
            messages.append(AIMessage(content=f"[{active.title()} Agent] Transferring you to our {target} specialist..."))
            active, handoffs = target, handoffs + 1
            continue

        messages.append(AIMessage(content=f"[{active.title()} Agent] {content}"))
        return {"messages": messages, "active_agent": active, "handoff_count": handoffs}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_query(query: str) -> None:
    print(f"Customer: {query}")
    result = run(query)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:300]}{'...' if len(msg.content) > 300 else ''}")
    print(f"  [Handoffs: {result['handoff_count']}, Final agent: {result['active_agent']}]\n")


def main():
    print("=== 09 Handoff Agent (Swarm Pattern, LangChain) ===\n")

    queries = [
        "I was charged twice on my last invoice. Can you help?",
        "I'm getting a 500 error when I try to log in to the dashboard.",
        "What are your business hours?",
    ]

    for query in queries:
        run_query(query)


if __name__ == "__main__":
    main()
