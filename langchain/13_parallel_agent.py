"""
13 - Parallel Agent (Map-Reduce via .batch(), LangChain)
Pattern: One analysis chain is mapped over every item concurrently with
         Runnable.batch(); a synthesis chain then reduces the results.
Use cases: Comparing many options, bulk document analysis, multi-source
           research, any embarrassingly parallel LLM workload.
LangGraph equivalent: ../13_parallel_agent.py (Send() fan-out + operator.add reducer)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
from typing import List
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

_analyze = ChatPromptTemplate.from_messages([
    ("system", "You are an expert analyst evaluating programming languages. "
               "Category to analyze: {category}. Be concise — 2-3 sentences max."),
    ("human", "Analyze '{item}' for {category}."),
]) | _llm | StrOutputParser()

_aggregate = ChatPromptTemplate.from_messages([
    ("system", "You are a synthesizer. Combine the individual analyses into a "
               "clear, comparative summary. Highlight key differences."),
    ("human", "Category: {category}\n\nIndividual analyses:\n{analyses}\n\nWrite a concise comparative summary:"),
]) | _llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Map (parallel) → reduce
# ---------------------------------------------------------------------------

def run(items: List[str], category: str) -> dict:
    # Map: .batch() runs one call per item concurrently (thread pool under the hood)
    outputs = _analyze.batch([{"item": item, "category": category} for item in items], config={"max_concurrency": 8})
    analyses = [f"[{item}] {out}" for item, out in zip(items, outputs)]

    # Reduce
    summary = _aggregate.invoke({"category": category, "analyses": "\n\n".join(analyses)})
    return {"analyses": analyses, "summary": summary}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== 13 Parallel Agent (Map-Reduce, LangChain) ===\n")

    items = ["Python", "Rust", "TypeScript", "Go"]
    category = "ecosystem maturity and tooling"
    print(f"Analyzing {len(items)} items in parallel: {items}")
    print(f"Category: {category}\n")

    result = run(items, category)

    print(f"Parallel analyses collected: {len(result['analyses'])}")
    print("\nIndividual Results:")
    for analysis in result["analyses"]:
        print(f"  {analysis[:150]}{'...' if len(analysis) > 150 else ''}")
    print(f"\nAggregated Summary:\n{result['summary']}")


if __name__ == "__main__":
    main()
