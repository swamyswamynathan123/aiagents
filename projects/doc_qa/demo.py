"""
Scripted demo of document Q&A over the sample Northwind documents.
Run from the repo root:  python -m projects.doc_qa.demo ["your question"]
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from projects.research_report.graph import format_cost  # noqa: E402

from .graph import DocQASession  # noqa: E402

CONVERSATION = [
    "How much paid time off do I get each year?",
    "And how many of those days can I carry over?",            # follow-up: needs the conversation
    "What does error E204 mean on the X200?",                   # exact code: keyword search matters
    "What address do I use to open the router's admin console?",  # conflicting versions: manual vs 3.2 notes
    "What is the company's policy on stock options?",           # not in the documents
]


def main():
    session = DocQASession()
    c = session.corpus
    print(f"Indexed {len(c.documents)} documents into {len(c.chunks)} chunks "
          f"(quarantined {len(c.quarantined)}, skipped {len(c.skipped)})\n")
    for question in sys.argv[1:] or CONVERSATION:
        r = session.ask(question)
        print(f"Q: {question}")
        if r["standalone"] and r["standalone"] != question:
            print(f"   ↳ searched as: {r['standalone']}")
        for h in r["hits"]:
            print(f"   📄 {h['label'][:60]:60} ranks={h['ranks']} relevance={h['relevance']}")
        print(f"A [{r['status']}]: {r['answer']}")
        for cit in r["citations"]:
            print(f"   [{cit['n']}] {cit['label']}")
        print()
    tokens = sum(u["input_tokens"] + u["output_tokens"] for u in session.usage.values())
    print(f"Tokens: {tokens:,} ({format_cost(session.usage)})")


if __name__ == "__main__":
    main()
