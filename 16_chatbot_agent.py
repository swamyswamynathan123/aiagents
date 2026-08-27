"""
16 - Chatbot Agent
Pattern: Basic multi-turn conversational agent. The simplest LangGraph
         pattern — one node, one edge, MemorySaver for conversation history.
         All other agent patterns in this project build on this foundation.
Use cases: Customer service bots, FAQ assistants, interview prep tools,
           onboarding bots, general-purpose conversational AI.

This is the canonical LangGraph "hello world" — start here if you're
learning the framework.

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# ---------------------------------------------------------------------------
# State — the simplest possible: just a message list
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# System prompt (customize per use case)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable AI assistant. "
    "You remember the full conversation history and refer back to it naturally. "
    "Keep responses concise (2-4 sentences) unless asked for more detail."
)

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def chat_node(state: ChatState) -> dict:
    resp = _llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"]))
    return {"messages": [resp]}


# ---------------------------------------------------------------------------
# Graph — minimal: one node, one edge
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()

_builder = StateGraph(ChatState)
_builder.add_node("chat", chat_node)
_builder.set_entry_point("chat")
_builder.add_edge("chat", END)
app = _builder.compile(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def chat(message: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Demo — simulated multi-turn conversation
# ---------------------------------------------------------------------------

def main():
    print("=== 16 Chatbot Agent ===\n")
    print("Simulating a multi-turn conversation (thread: 'demo-1')\n")

    turns = [
        "Hi! My name is Jordan. What can you help me with?",
        "I'm learning about AI agents. Can you give me a quick overview?",
        "What's the difference between a ReAct agent and a Planning agent?",
        "What was my name again? (Testing memory)",
    ]

    for user_input in turns:
        print(f"User : {user_input}")
        response = chat(user_input, thread_id="demo-1")
        print(f"Bot  : {response}")
        print()

    # Interactive mode — uncomment to enable live chat
    # print("\n--- Interactive Mode (type 'quit' to exit) ---")
    # while True:
    #     user_input = input("You: ").strip()
    #     if user_input.lower() in ("quit", "exit", "q"):
    #         break
    #     response = chat(user_input, thread_id="interactive-1")
    #     print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
