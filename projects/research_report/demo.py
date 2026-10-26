"""
Scripted demo of the research report generator, with live progress.
Run from the repo root:  python -m projects.research_report.demo ["your topic"]
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from .graph import ReportSession, format_cost  # noqa: E402


def progress(node: str, update: dict) -> None:
    if node == "planner":
        print(f"📝 plan: {len(update['plan'])} questions")
        for q in update["plan"]:
            print(f"     - {q['question']}")
    elif node == "research":
        f = update["findings"][0]
        print(f"🔎 researched: {f['question'][:70]} — {len(f['facts'])} facts")
        for b in update.get("blocked_sources", []):
            print(f"     🛡️ dropped untrusted source {b['url']} ({b['reason']})")
    elif node == "writer":
        print(f"✍️  draft written ({len(update['draft'].split())} words, {len(update['references'])} sources)")
    elif node == "reviewer":
        r = update["reviews"][-1]
        print(f"🧐 review: {r['score']}/10 → {r['decision']}" + (f" (gaps: {r['gaps']})" if r["gaps"] else ""))
    elif node == "finalize":
        print("✅ finalized")


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "The current state of solid-state batteries for electric vehicles"
    print(f"=== Research Report: {topic} ===\n")
    session = ReportSession()
    for node, update in session.stream(topic, audience="technology-curious general readers"):
        progress(node, update)

    result = session.result
    print(f"\nStatus: {result['status']} | scores {result['scores']} | revisions {result['revisions']} | "
          f"research rounds {result['research_rounds']} | sources {len(result['references'])}")
    tokens = sum(u["input_tokens"] + u["output_tokens"] for u in result["usage"].values())
    print(f"Tokens: {tokens:,} ({format_cost(result['usage'])})\n")
    print(result["report"])


if __name__ == "__main__":
    main()
