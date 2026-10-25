"""
14 - RAG Agent (Retrieval-Augmented Generation, LangChain)
Pattern: Classic LCEL RAG chain — retriever → format context → prompt →
         llm → parser — with RunnablePassthrough.assign() so the retrieved
         documents are returned alongside the answer.
Use cases: Q&A over internal docs, support bots grounded in a knowledge
           base, any answer that must cite a known corpus.
LangGraph equivalent: ../14_rag_agent.py (retrieve → generate graph)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
    "LangChain Expression Language (LCEL) composes runnables with the | operator: prompt | llm | parser.",
    "RunnableWithMessageHistory adds per-session chat history to any LangChain runnable without a graph.",
]

# ---------------------------------------------------------------------------
# Vector store (in-memory FAISS index, built on first use)
# ---------------------------------------------------------------------------

_vectorstore: Optional[FAISS] = None


def _build_vectorstore() -> FAISS:
    print("Building vector store (this calls OpenAI Embeddings API)...")
    vs = FAISS.from_texts(KNOWLEDGE_BASE, embedding=_embeddings)
    print(f"Vector store ready: {len(KNOWLEDGE_BASE)} documents indexed.\n")
    return vs

# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable assistant. Answer the user's question "
               "using ONLY the provided context. If the context doesn't contain "
               "enough information, say so explicitly.\n\nContext:\n{context}"),
    ("human", "{query}"),
])
_answer_chain = (
    RunnablePassthrough.assign(context=lambda x: "\n".join(f"- {d}" for d in x["retrieved_docs"]))
    | _prompt | _llm | StrOutputParser()
)


def run(query: str, num_results: int = 4) -> dict:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    retriever = _vectorstore.as_retriever(search_kwargs={"k": num_results})

    chain = (
        RunnablePassthrough.assign(retrieved_docs=(lambda x: x["query"]) | retriever | (lambda docs: [d.page_content for d in docs]))
        | RunnablePassthrough.assign(answer=_answer_chain)
    )
    return chain.invoke({"query": query})


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def ask(query: str, num_results: int = 4) -> None:
    print(f"Q: {query}")
    result = run(query, num_results)
    print(f"Retrieved {len(result['retrieved_docs'])} docs:")
    for doc in result["retrieved_docs"]:
        print(f"  • {doc[:100]}{'...' if len(doc) > 100 else ''}")
    print(f"A: {result['answer']}\n")


def main():
    print("=== 14 RAG Agent (LangChain) ===\n")

    questions = [
        "What is LangGraph and how does it work?",
        "How does LCEL compose a chain?",
        "What is LangSmith used for?",
        "Explain the difference between Supervisor and Swarm multi-agent patterns.",
    ]

    for question in questions:
        ask(question)


if __name__ == "__main__":
    main()
