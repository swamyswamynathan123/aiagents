"""
Scripted demo of the data analyst on the sample sales dataset.
Run from the repo root:  python -m projects.data_analyst.demo ["your question"]
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from projects.research_report.graph import format_cost  # noqa: E402

from .graph import AnalystSession  # noqa: E402

QUESTIONS = [
    "Which region had the highest total revenue in 2025, and what share of the total was it?",
    "Plot monthly revenue by category and tell me which category is the most seasonal.",
    "What's our customer satisfaction score by region?",
]


_seen = {"runs": 0}


def progress(node: str, update: dict) -> None:
    if node == "input_guard":
        _seen["runs"] = 0
    if node == "understand" and update.get("spec"):
        s = update["spec"]
        print(f"  🧭 plan: {s['intent']} · metrics {s['metrics']} · by {s['dimensions']} · chart {s['needs_chart']}")
    elif node == "executor":
        for n, e in enumerate(update.get("executions", [])[_seen["runs"]:], _seen["runs"] + 1):
            first = e["code"].strip().splitlines()[0][:70] if e["code"].strip() else ""
            print(f"  🐍 run #{n}: {first}… → {'ok' if e['ok'] else 'error'}")
        _seen["runs"] = len(update.get("executions", []))
    elif node == "verify":
        v = update["verification"]
        print(f"  🔍 verify: ungrounded {v['ungrounded'] or 'none'} · critic {'✓' if v['critic_approved'] else '✗'}"
              + (" → revising" if update.get("revisions") else ""))


def main():
    session = AnalystSession()
    for question in sys.argv[1:] or QUESTIONS:
        print(f"Q: {question}")
        for node, update in session.stream(question):
            progress(node, update)
        r = session.result
        print(f"  [{r['status']}] charts: {r['charts'] or 'none'}")
        print(f"A: {r['answer']}\n")
    tokens = sum(u["input_tokens"] + u["output_tokens"] for u in session.usage.values())
    print(f"Tokens: {tokens:,} ({format_cost(session.usage)})")


if __name__ == "__main__":
    main()
