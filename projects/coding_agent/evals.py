"""
Evals for the coding agent (real API). Each task is a known bug, so grading is objective:
  - fixed            : the visible test suite passes
  - hidden_pass      : the HIDDEN tests pass too (catches fixes that only special-case visible tests)
  - tests_untouched  : no test file was changed
  - changed_lines    : the diff stays small (catches sprawling rewrites)
  - no_regressions   : no edit was kept that increased failing tests (guaranteed by the rollback)

Run from the repo root:
  python -m projects.coding_agent.evals            # LangSmith experiment + local summary
  python -m projects.coding_agent.evals --local    # nothing uploaded
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "evals")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


def call(ctx, inputs):
    from . import workspace as ws
    from .graph import CodingSession, build_app
    from .tasks import with_hidden_tests

    session = CodingSession(inputs["task"], graph=build_app())
    r = session.run()
    hidden = ws.run_tests(with_hidden_tests(session.workspace))
    tests_changed = [p for p in ws.snapshot(session.workspace) if ws.is_test_file(p)
                     and ws.snapshot(session.workspace)[p] != session.graph.get_state(session.config).values["original"].get(p)]
    return {
        "answer": r["summary"],
        "tools": r["tool_calls"],
        "fields": {
            "status": r["status"],
            "hidden_pass": hidden["ok"],
            "tests_untouched": not tests_changed,
            "changed_lines": ws.changed_lines(r["diff"]),
            "steps": r["steps"],
        },
    }


# Outcomes only: tests rerun after every edit, so requiring an explicit run_tests call would grade
# the agent's process rather than its result.
FIXED = {"fields": {"status": "fixed", "hidden_pass": True, "tests_untouched": True},
         "must_use_tools": ["edit_file"]}

SUITE = {
    "call": call,
    "examples": [
        {"inputs": {"task": "tier-boundary"}, "reference": {**FIXED, "max_fields": {"changed_lines": 4}}},
        {"inputs": {"task": "discount-percent"}, "reference": {**FIXED, "max_fields": {"changed_lines": 4}}},
        {"inputs": {"task": "remove-exact"}, "reference": {**FIXED, "max_fields": {"changed_lines": 4}}},
        {"inputs": {"task": "rounding"}, "reference": {**FIXED, "max_fields": {"changed_lines": 6}}},
        {"inputs": {"task": "oversell"}, "reference": {**FIXED, "max_fields": {"changed_lines": 6}}},
        {"inputs": {"task": "missing-clear"}, "reference": {**FIXED, "max_fields": {"changed_lines": 6}}},
        {"inputs": {"task": "two-bugs"}, "reference": {**FIXED, "max_fields": {"changed_lines": 10}}},
    ],
}


def main():
    from run_evals import _print_rows, run_suite

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    upload = not args.local and bool(os.getenv("LANGCHAIN_API_KEY"))
    if not upload:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    rows = run_suite(SUITE, {}, "coding-agent", upload=upload, use_judge=False,
                     prefix="coding-agent", metadata={"project": "coding_agent"}, concurrency=2)
    failures = _print_rows("coding agent tasks", rows)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
