"""
03 - Router Agent (LangChain)
Pattern: A structured-output classifier picks a route; RunnableLambda
         dispatches to the matching specialist chain (LCEL routing).
Use cases: Customer support triage, multi-domain assistants, cost-optimized
           pipelines that send easy queries to cheaper chains.
LangGraph equivalent: ../03_router_agent.py (add_conditional_edges)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Routing schema + classifier chain
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    route: Literal["math", "writing", "general"]
    reason: str


_classifier = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user query into exactly one category:\n"
     "- math: arithmetic, algebra, statistics, numbers, calculations\n"
     "- writing: essays, poems, summaries, creative writing, editing\n"
     "- general: anything else (history, science, trivia, advice)"),
    ("human", "{query}"),
]) | _llm.with_structured_output(RouteDecision)

# ---------------------------------------------------------------------------
# Specialist chains
# ---------------------------------------------------------------------------

def _specialist(system_prompt: str):
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")]) | _llm | StrOutputParser()


SPECIALISTS = {
    "math": _specialist("You are a precise math tutor. Show your work step by step."),
    "writing": _specialist("You are a creative writing expert. Be vivid and engaging."),
    "general": _specialist("You are a knowledgeable general assistant. Be concise and accurate."),
}

# ---------------------------------------------------------------------------
# Router chain — classify, then dispatch
# ---------------------------------------------------------------------------

_router_chain = (
    RunnablePassthrough.assign(route=_classifier | (lambda d: d.route))
    | RunnablePassthrough.assign(response=RunnableLambda(lambda x: SPECIALISTS[x["route"]]))
)


def run(query: str) -> dict:
    return _router_chain.invoke({"query": query})


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 03 Router Agent (LangChain) ===\n")

    test_queries = [
        "What is the square root of 144 plus 2 to the power of 8?",
        "Write a haiku about the Python programming language.",
        "Who invented the World Wide Web and when?",
    ]

    for query in test_queries:
        result = run(query)
        print(f"Query : {query}")
        print(f"Route : {result['route']}")
        print(f"Answer: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
