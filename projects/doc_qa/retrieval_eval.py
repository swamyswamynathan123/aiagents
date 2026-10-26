"""
Retrieval benchmark: how often does each search mode find the right passage?

Each labelled question names the passage that answers it by a phrase that passage contains
(robust to re-chunking). For every mode it reports:
  hit@1     — the right passage is ranked first
  recall@k  — the right passage is somewhere in the top k
  MRR       — mean reciprocal rank (1/rank of the right passage, 0 if missed)
This measures retrieval on its own, separately from answer quality: most RAG failures are
retrieval failures, and they are invisible if you only grade final answers.

Run from the repo root:  python -m projects.doc_qa.retrieval_eval [--k 4] [--modes bm25,vector]
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# (question, phrase in the gold passage, what the question tests)
LABELLED = [
    ("What does error E204 mean?", "E204 - WAN link down", "exact code"),
    ("Router shows code E415, what's wrong?", "E415 - Too many devices", "exact code"),
    ("How much vacation do I earn every month?", "accrue 1.5 days", "paraphrase (vacation → PTO)"),
    ("Can I roll unused holidays into next year?", "carry over into the next", "paraphrase (roll over → carryover)"),
    ("How long is maternity leave?", "16 weeks of fully paid", "paraphrase (maternity → parental)"),
    ("What is the daily meal allowance on business trips?", "$75 per day", "paraphrase"),
    ("Do I need a receipt for a $40 taxi?", "Receipts are required", "reasoning over a threshold"),
    ("How do I wipe the router back to factory settings?", "Factory reset: hold", "paraphrase (wipe → factory reset)"),
    ("What does a blinking red light mean?", "Blinking red", "lookup"),
    ("Where do I report a phishing email?", "security@northwind.example", "lookup"),
    ("Is WPA3-only mode supported?", "WPA3-only mode", "exact feature name"),
    ("How often do I get a new laptop?", "refreshed every 3 years", "paraphrase"),
    ("What is the minimum password length?", "at least 14 characters", "lookup"),
    ("When do automatic firmware updates run?", "Sundays at 3:00 am", "lookup"),
    ("How many days a week can I work from home?", "up to 3 days per week", "paraphrase (from home → remotely)"),
    ("What is the warranty on the X200?", "2 years limited hardware warranty", "lookup"),
]


def rank_of(hits, phrase: str):
    for n, h in enumerate(hits, 1):
        if phrase.lower() in h.chunk.text.lower():
            return n
    return None


def evaluate(retriever, modes, k: int):
    rows = {}
    for mode in modes:
        ranks = [rank_of(retriever.search(q, mode=mode, k=k), phrase) for q, phrase, _ in LABELLED]
        rows[mode] = {
            "hit@1": sum(r == 1 for r in ranks) / len(ranks),
            f"recall@{k}": sum(r is not None for r in ranks) / len(ranks),
            "MRR": sum(1 / r for r in ranks if r) / len(ranks),
            "misses": [LABELLED[i][0] for i, r in enumerate(ranks) if r is None],
        }
    return rows


def main():
    from .graph import SAMPLE_DOCS
    from .ingest import ingest
    from .retrieval import MODES, Retriever

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()

    retriever = Retriever(ingest(sorted(Path(SAMPLE_DOCS).iterdir())))
    rows = evaluate(retriever, args.modes.split(","), args.k)
    print(f"{len(LABELLED)} labelled questions, {len(retriever.corpus.chunks)} chunks, k={args.k}\n")
    print(f"{'mode':16} {'hit@1':>7} {'recall@' + str(args.k):>9} {'MRR':>6}")
    for mode, m in rows.items():
        print(f"{mode:16} {m['hit@1']:7.0%} {m[f'recall@{args.k}']:9.0%} {m['MRR']:6.2f}"
              + (f"   missed: {m['misses']}" if m["misses"] else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
