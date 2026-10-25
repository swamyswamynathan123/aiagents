"""
16 - Chatbot Agent (LangChain)
Pattern: The simplest stateful LangChain app — prompt | llm wrapped in
         RunnableWithMessageHistory, keyed by session_id.
Use cases: Conversational assistants, FAQ bots, onboarding helpers, the
           starting point before adding tools or retrieval.
LangGraph equivalent: ../16_chatbot_agent.py (one-node graph + MemorySaver)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import Dict
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# System prompt (customize per use case)
SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable AI assistant. "
    "You remember the full conversation history and refer back to it naturally. "
    "Keep responses concise (2-4 sentences) unless asked for more detail."
)

# ---------------------------------------------------------------------------
# Chain with per-session history
# ---------------------------------------------------------------------------

_histories: Dict[str, InMemoryChatMessageHistory] = {}

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

_chain = RunnableWithMessageHistory(
    _prompt | _llm,
    lambda session_id: _histories.setdefault(session_id, InMemoryChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history",
)


def chat(message: str, thread_id: str = "default") -> str:
    return _chain.invoke({"input": message}, config={"configurable": {"session_id": thread_id}}).content


# ---------------------------------------------------------------------------
# Demo — simulated multi-turn conversation
# ---------------------------------------------------------------------------

def main():
    print("=== 16 Chatbot Agent (LangChain) ===\n")
    print("Simulating a multi-turn conversation (session: 'demo-1')\n")

    turns = [
        "Hi! My name is Jordan. What can you help me with?",
        "I'm learning about AI agents. Can you give me a quick overview?",
        "What's the difference between a ReAct agent and a Planning agent?",
        "What was my name again? (Testing memory)",
    ]

    for user_input in turns:
        print(f"User : {user_input}")
        print(f"Bot  : {chat(user_input, thread_id='demo-1')}")
        print()


if __name__ == "__main__":
    main()
