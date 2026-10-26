"""
14 - RAG Agent (Retrieval-Augmented Generation)
Pattern: Retrieve relevant documents from a vector store before generating
         an answer. The LLM only sees context that was retrieved — reducing
         hallucination and grounding responses in real documents.
Use cases: Document Q&A, knowledge bases, customer support with product docs,
           legal/medical document analysis, internal wiki assistants.

Dependencies: langchain-community, faiss-cpu (included in requirements.txt)

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# Sample knowledge base (in a real app these come from files or a DB)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    "LangGraph is a library for building stateful, multi-actor AI applications using graphs.",
    "LangGraph uses StateGraph as its core abstraction. Nodes represent computation, edges represent control flow.",
    "LangGraph supports cyclic graphs, enabling agents to loop and retry actions until a goal is met.",
    "LangSmith is an observability platform for LLM applications. It traces every node and tool call in LangGraph.",
    "LangSmith enables debugging, evaluation, and monitoring of AI agents in production.",
    "LangChain provides LLM abstractions including ChatOpenAI, tools, and prompt templates.",
    "LangGraph's MemorySaver checkpointer persists conversation state across multiple invocations.",
    "The ReAct pattern (Reason + Act) is a popular agent architecture: the LLM reasons, selects a tool, observes the result, then repeats.",
    "Human-in-the-loop in LangGraph uses interrupt() to pause execution and Command(resume=value) to continue.",
    "LangGraph's Send() primitive enables fan-out: dispatching the same node in parallel for multiple inputs.",
    "Multi-agent systems in LangGraph can use a Supervisor pattern (central coordinator) or Swarm pattern (peer-to-peer handoff).",
    "LangGraph supports structured output via Pydantic models and the with_structured_output() method on LLMs.",
    "RAG (Retrieval-Augmented Generation) grounds LLM answers in retrieved documents, reducing hallucination.",
    "FAISS is a library for efficient vector similarity search, commonly used for in-memory RAG implementations.",
    "OpenAI's text-embedding-3-small model produces 1536-dimensional embeddings suitable for semantic search.",
    "LangGraph graphs are compiled before use: graph = builder.compile(checkpointer=..., store=...)",
    "Pydantic v2 models are used with LangGraph for type-safe state definitions and structured output schemas.",
    "The planning agent pattern (Plan-and-Execute) generates a multi-step plan upfront, then executes each step.",
    "Reflection agents use a generator-critic loop: one LLM generates content, another evaluates and requests revisions.",
    "Autonomous agents in LangGraph self-direct their tool use, guided only by a high-level goal and a max_iterations cap.",
]

# ---------------------------------------------------------------------------
# Build vector store (in-memory FAISS index)
# ---------------------------------------------------------------------------

def _build_vectorstore() -> FAISS:
    print("Building vector store (this calls OpenAI Embeddings API)...")
    vs = FAISS.from_texts(KNOWLEDGE_BASE, embedding=_embeddings)
    print(f"Vector store ready: {len(KNOWLEDGE_BASE)} documents indexed.\n")
    return vs


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[str]
    answer: str
    num_results: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_vectorstore: Optional[FAISS] = None  # lazy-initialized in main()


def retrieve_node(state: RAGState) -> dict:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    results = _vectorstore.similarity_search(state["query"], k=state["num_results"])
    docs = [doc.page_content for doc in results]
    return {"retrieved_docs": docs}


def generate_node(state: RAGState) -> dict:
    context = "\n".join(f"- {doc}" for doc in state["retrieved_docs"])
    resp = _llm.invoke([
        SystemMessage(content=(
            "You are a knowledgeable assistant. Answer the user's question "
            "using ONLY the provided context. If the context doesn't contain "
            "enough information, say so explicitly."
            f"\n\nContext:\n{context}"
        )),
        HumanMessage(content=state["query"]),
    ])
    return {"answer": resp.content}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(RAGState)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("generate", generate_node)
_builder.set_entry_point("retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def ask(query: str, num_results: int = 4) -> None:
    print(f"Q: {query}")
    result = app.invoke({"query": query, "retrieved_docs": [], "answer": "", "num_results": num_results})
    print(f"Retrieved {len(result['retrieved_docs'])} docs:")
    for doc in result["retrieved_docs"]:
        print(f"  • {doc[:100]}{'...' if len(doc) > 100 else ''}")
    print(f"A: {result['answer']}\n")


def main():
    global _vectorstore
    print("=== 14 RAG Agent (Retrieval-Augmented Generation) ===\n")
    _vectorstore = _build_vectorstore()

    questions = [
        "What is LangGraph and how does it work?",
        "How does the human-in-the-loop pattern work in LangGraph?",
        "What is LangSmith used for?",
        "Explain the difference between Supervisor and Swarm multi-agent patterns.",
    ]

    for question in questions:
        ask(question)


if __name__ == "__main__":
    main()
