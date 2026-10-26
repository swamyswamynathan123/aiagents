"""
Scripted demo of the coding agent.
Run from the repo root:  python -m projects.coding_agent.demo [task-id ...]
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from projects.research_report.graph import format_cost  # noqa: E402

from . import workspace as ws  # noqa: E402
from .graph import CodingSession  # noqa: E402
from .tasks import TASKS, with_hidden_tests  # noqa: E402


def progress(node: str, update: dict) -> None:
    if node == "baseline":
        b = update["baseline"]
        print(f"  🧪 baseline: {b['passed']} passed, {ws.failing_count(b)} failing {b['failing']}")
    elif node == "agent":
        for m in update.get("messages", []):
            for c in getattr(m, "tool_calls", None) or []:
                arg = c["args"].get("path") or c["args"].get("query") or c["args"].get("target") or ""
                print(f"  🔧 {c['name']}({arg})")
    elif node == "verify" and "final_tests" in update:
        f = update["final_tests"]
        print(f"  ✅ verify: {f['passed']} passed, {ws.failing_count(f)} failing")


def main():
    task_ids = sys.argv[1:] or ["tier-boundary", "two-bugs"]
    for task_id in task_ids:
        print(f"=== {task_id}: {TASKS[task_id]['title']} ===")
        session = CodingSession(task_id)
        for node, update in session.stream():
            progress(node, update)
        r = session.result
        hidden = ws.run_tests(with_hidden_tests(session.workspace))
        print(f"  [{r['status']}] steps {r['steps']}, test runs {r['test_runs']}, "
              f"edits {[(e['path'], e['outcome']) for e in r['edits']]}")
        print(f"  hidden tests: {hidden['passed']} passed, {ws.failing_count(hidden)} failing")
        print(f"  summary (from the diff): {r['summary']}")
        if r["agent_summary"] and r["agent_summary"] != r["summary"]:
            print(f"  agent said: {r['agent_summary']}")
        for e in r["rejected_test_edits"]:
            print(f"  🛡️ rejected an attempt to edit {e['path']}: tests were not modified")
        print(r["diff"] or "  (no changes)")
        print(f"  tokens: {sum(u['input_tokens'] + u['output_tokens'] for u in r['usage'].values()):,} "
              f"({format_cost(r['usage'])})\n")


if __name__ == "__main__":
    main()
