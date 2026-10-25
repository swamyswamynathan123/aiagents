"""
06 - Memory Agent (LangChain)
Pattern: RunnableWithMessageHistory keeps per-session conversation history
         (short-term memory); a plain dict of facts per user acts as
         long-term memory injected into the prompt on every turn.
Use cases: Personal assistants, customer support with history, tutoring
           bots that remember a learner across sessions.
LangGraph equivalent: ../06_memory_agent.py (MemorySaver + InMemoryStore)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import re
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

_FACT_PATTERNS = [
    (re.compile(r"\bmy name is (\w+)", re.IGNORECASE), "User's name is {}"),
    (re.compile(r"\bi (?:love|like|enjoy) (\w[\w\s]+?)(?:\.|,|$)", re.IGNORECASE), "User likes {}"),
    (re.compile(r"\bi (?:work|am) (?:as |a |an )?([a-zA-Z\s]+?)(?:\.|,|$)", re.IGNORECASE), "User works as {}"),
    (re.compile(r"\bi(?:'m| am) from ([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.IGNORECASE), "User is from {}"),
]

# ---------------------------------------------------------------------------
# Memory stores (in-process; swap for Redis/SQL-backed histories in production)
# ---------------------------------------------------------------------------

_histories: Dict[str, InMemoryChatMessageHistory] = {}  # short-term: per session
_user_facts: Dict[str, List[str]] = {}                  # long-term: per user


def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    return _histories.setdefault(session_id, InMemoryChatMessageHistory())


def _extract_and_save_facts(user_id: str, text: str) -> None:
    """Heuristically extract simple facts from user message and save them."""
    facts = _user_facts.setdefault(user_id, [])
    for pattern, template in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            fact = template.format(match.group(1).strip())
            if fact not in facts:
                facts.append(fact)

# ---------------------------------------------------------------------------
# Chain with history
# ---------------------------------------------------------------------------

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful, personalized AI assistant.\n"
     "Known facts about this user:\n{facts}\n\n"
     "If the user mentions new personal facts (name, preferences, goals), "
     "acknowledge them naturally. Be warm and use their name if known."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

_chain = RunnableWithMessageHistory(
    _prompt | _llm,
    _get_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat(user_id: str, thread_id: str, message: str) -> str:
    facts = _user_facts.get(user_id, [])
    response = _chain.invoke(
        {"input": message, "facts": "\n".join(f"- {f}" for f in facts) or "None yet."},
        config={"configurable": {"session_id": thread_id}},
    )
    _extract_and_save_facts(user_id, message)
    return response.content


# ---------------------------------------------------------------------------
# Demo (multi-turn conversation with memory)
# ---------------------------------------------------------------------------

def main():
    print("=== 06 Memory Agent (LangChain) ===\n")

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

    # A brand-new session: history is empty, but long-term facts still apply
    print("--- New session (different thread, same user) ---\n")
    response = chat(user_id, "alice-session-2", "Do you remember my name?")
    print("User : Do you remember my name?")
    print(f"Agent: {response[:300]}")


if __name__ == "__main__":
    main()
