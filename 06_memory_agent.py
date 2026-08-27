"""
06 - Memory Agent
Pattern: MemorySaver for per-thread conversation history (short-term memory)
         + InMemoryStore for cross-thread user facts (long-term memory).
Use cases: Persistent chatbots, personalized assistants, multi-session agents
           that need to remember user preferences across conversations.

Architecture:
  - MemorySaver (checkpointer): saves the full message history per thread_id
  - InMemoryStore: key-value store for cross-thread facts (name, preferences, etc.)

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import re
import uuid
from typing import Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

_FACT_PATTERNS = [
    (re.compile(r"\bmy name is (\w+)", re.IGNORECASE), "User's name is {}"),
    (re.compile(r"\bi (?:love|like|enjoy) (\w[\w\s]+?)(?:\.|,|$)", re.IGNORECASE), "User likes {}"),
    (re.compile(r"\bi (?:work|am) (?:as |a |an )?([a-zA-Z\s]+?)(?:\.|,|$)", re.IGNORECASE), "User works as {}"),
    (re.compile(r"\bi(?:'m| am) from ([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.IGNORECASE), "User is from {}"),
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MemoryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str


# ---------------------------------------------------------------------------
# Store + checkpointer
# ---------------------------------------------------------------------------

_store = InMemoryStore()
_checkpointer = MemorySaver()


# ---------------------------------------------------------------------------
# Nodes
# In LangGraph 1.x, nodes access the store via the `store` parameter.
# ---------------------------------------------------------------------------

def recall_and_respond(state: MemoryState, *, store: InMemoryStore) -> dict:
    user_id = state["user_id"]

    # Retrieve persisted facts for this user
    facts_items = store.search(("user_facts", user_id))
    facts_text = "\n".join(f"- {item.value['fact']}" for item in facts_items) if facts_items else "None yet."

    system_prompt = (
        f"You are a helpful, personalized AI assistant.\n"
        f"Known facts about this user:\n{facts_text}\n\n"
        "If the user mentions new personal facts (name, preferences, goals), "
        "acknowledge them naturally. Be warm and use their name if known."
    )

    resp = _llm.invoke([SystemMessage(content=system_prompt)] + list(state["messages"]))

    # Extract and persist any new facts mentioned in the latest human message
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if last_human:
        _extract_and_save_facts(store, user_id, last_human.content, facts_items)

    return {"messages": [resp]}


def _extract_and_save_facts(store: InMemoryStore, user_id: str, text: str, existing) -> None:
    """Heuristically extract simple facts from user message and save to store."""
    existing_facts = {item.value["fact"] for item in existing}
    for pattern, template in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            fact = template.format(match.group(1).strip())
            if fact not in existing_facts:
                store.put(("user_facts", user_id), str(uuid.uuid4()), {"fact": fact})


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(MemoryState)
_builder.add_node("respond", recall_and_respond)
_builder.set_entry_point("respond")
_builder.add_edge("respond", END)
app = _builder.compile(checkpointer=_checkpointer, store=_store)


# ---------------------------------------------------------------------------
# Demo (multi-turn conversation with memory)
# ---------------------------------------------------------------------------

def chat(user_id: str, thread_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=message)], "user_id": user_id},
        config=config,
    )
    return result["messages"][-1].content


def main():
    print("=== 06 Memory Agent ===\n")
    user_id = "alice"
    thread_id = "alice-session-1"

    turns = [
        "Hi! My name is Alice and I love Python programming.",
        "I work as a data scientist in San Francisco.",
        "What do you know about me so far?",
        "Can you recommend a Python library for data visualization?",
    ]

    for message in turns:
        print(f"User : {message}")
        response = chat(user_id, thread_id, message)
        print(f"Agent: {response[:300]}{'...' if len(response) > 300 else ''}")
        print()

    # Second session — MemorySaver restores conversation history automatically
    print("--- New session (same thread) ---\n")
    response = chat(user_id, thread_id, "What was the first thing I told you?")
    print(f"User : What was the first thing I told you?")
    print(f"Agent: {response[:300]}")


if __name__ == "__main__":
    main()
