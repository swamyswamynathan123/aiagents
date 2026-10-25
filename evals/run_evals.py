"""
Eval runner — runs the cases in evals/cases.py against the LangGraph and/or
LangChain versions of each agent, plus the guardrails red-team suite.

With LANGCHAIN_API_KEY set, each suite becomes a LangSmith dataset and each run
an experiment (prefix "<framework>-<agent>"), so you can compare frameworks in
the LangSmith UI. --local runs everything in-process with no uploads or tracing.
A pass/fail summary is always printed; exit code is 1 if any example fails.

Usage:
  python evals/run_evals.py                          # everything, both frameworks
  python evals/run_evals.py --agents 03,11           # selected agents
  python evals/run_evals.py --framework langchain    # one framework
  python evals/run_evals.py --guardrails-only        # just the red-team suite
  python evals/run_evals.py --local --no-judge       # offline-ish: no uploads, no LLM judge
"""
import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DATASET_PREFIX = "aiagents"
JUDGE_PASS = 4  # LLM judge score (1-5) needed to pass

# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_judge = None


def llm_judge(question: str, answer: str, criteria: str):
    global _judge
    if _judge is None:
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field

        class Verdict(BaseModel):
            score: int = Field(description="1 (fails the criteria) to 5 (fully meets them)", ge=1, le=5)
            reasoning: str = Field(description="One or two sentences")

        _judge = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Verdict)
    v = _judge.invoke([
        ("system", f"You are a strict evaluator. Today's date is {date.today().isoformat()}. "
                   "Score how well the answer meets the criteria. The criteria define correct behavior: "
                   "judge only against them, not against your own knowledge of the right answer or style. "
                   "Treat dates up to today as current, not future."),
        ("human", f"Input:\n{question}\n\nAnswer:\n{answer}\n\nCriteria:\n{criteria}"),
    ])
    return v.score, v.reasoning


def _norm(s):
    return str(s).lower().replace(",", "")


def make_checks(use_judge: bool):
    def checks(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        ref, results = reference_outputs or {}, []
        if outputs.get("error"):
            return {"results": [{"key": "error", "score": 0, "comment": outputs["error"][:500]}]}
        answer, tools, fields = outputs.get("answer", ""), outputs.get("tools", []), outputs.get("fields", {})

        for s in ref.get("must_contain", []):
            results.append({"key": f"contains:{s[:20]}", "score": int(_norm(s) in _norm(answer))})
        for s in ref.get("must_not_contain", []):
            results.append({"key": f"excludes:{s[:20]}", "score": int(_norm(s) not in _norm(answer))})
        for t in ref.get("must_use_tools", []):
            results.append({"key": f"tool:{t}", "score": int(t in tools), "comment": f"called: {tools}"})
        if ref.get("any_tool_of"):
            results.append({"key": "tool:any", "score": int(any(t in tools for t in ref["any_tool_of"])), "comment": f"called: {tools}"})
        for k, want in ref.get("fields", {}).items():
            got = fields.get(k)
            ok = _norm(got) == _norm(want) if isinstance(want, str) else got == want
            results.append({"key": f"field:{k}", "score": int(ok), "comment": f"expected {want!r}, got {got!r}"})
        for k, n in ref.get("min_fields", {}).items():
            results.append({"key": f"min:{k}", "score": int((fields.get(k) or 0) >= n), "comment": f"{fields.get(k)} >= {n}"})
        for k, n in ref.get("max_fields", {}).items():
            results.append({"key": f"max:{k}", "score": int((fields.get(k) or 0) <= n), "comment": f"{fields.get(k)} <= {n}"})
        if ref.get("criteria") and use_judge:
            question = json.dumps(inputs, default=str)[:2000]
            score, reasoning = llm_judge(question, answer, ref["criteria"])
            results.append({"key": "judge", "score": int(score >= JUDGE_PASS), "comment": f"{score}/5 — {reasoning}"})
        return {"results": results}
    return checks


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _suite_hash(suite) -> str:
    return hashlib.sha1(json.dumps(suite["examples"], sort_keys=True).encode()).hexdigest()[:8]


def sync_dataset(client, name: str, suite) -> str:
    """One dataset per suite, versioned by content hash so edited cases get a fresh dataset."""
    full = f"{DATASET_PREFIX}-{name}-{_suite_hash(suite)}"
    if not client.has_dataset(dataset_name=full):
        ds = client.create_dataset(full, description=f"Eval cases for {name} (from evals/cases.py)")
        client.create_examples(
            inputs=[e["inputs"] for e in suite["examples"]],
            outputs=[e["reference"] for e in suite["examples"]],
            dataset_id=ds.id,
        )
    return full


def local_examples(suite):
    from langsmith.schemas import Example

    ds_id = uuid.uuid4()
    return [Example(id=uuid.uuid4(), dataset_id=ds_id, inputs=e["inputs"], outputs=e["reference"]) for e in suite["examples"]]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def make_target(suite, ctx):
    def target(inputs: dict) -> dict:
        try:
            return suite["call"](ctx, inputs)
        except Exception as exc:  # record failures as outputs so the evaluator can report them
            return {"error": f"{type(exc).__name__}: {exc}"}
    return target


def run_suite(suite, ctx, name: str, *, upload: bool, use_judge: bool, prefix: str, metadata: dict, concurrency=2):
    """Run one suite; returns a list of row dicts for the summary."""
    from langsmith import Client, evaluate

    data = sync_dataset(Client(), name, suite) if upload else local_examples(suite)
    results = evaluate(
        make_target(suite, ctx),
        data=data,
        evaluators=[make_checks(use_judge)],
        experiment_prefix=prefix,
        metadata=metadata,
        max_concurrency=concurrency,
        upload_results=upload,
    )
    rows = []
    for r in results:
        checks = [(e.key, e.score, e.comment) for e in r["evaluation_results"]["results"]]
        rows.append({
            "inputs": r["example"].inputs,
            "outputs": r["run"].outputs or {},
            "checks": checks,
            "passed": bool(checks) and all(s for _, s, _ in checks),
        })
    return rows


def run_agent_evals(framework: str, num: str, module=None, *, upload=False, use_judge=True):
    """Evaluate one agent in one framework (used by the CLI and the Streamlit Evals tab)."""
    from cases import SUITES
    from registry import load_module, make_runner, module_path

    path = module_path(framework, num)
    module = module or load_module(path)
    ctx = {"framework": framework, "module": module, "run": make_runner(framework, num, module)}
    return run_suite(
        SUITES[num], ctx, f"{num}-{path.stem[3:]}",
        upload=upload, use_judge=use_judge,
        prefix=f"{framework.lower()}-{num}", metadata={"framework": framework, "agent": path.stem},
    )


def run_guardrail_evals(*, upload=False):
    from cases import GUARDRAIL_SUITE

    return run_suite(GUARDRAIL_SUITE, {}, "guardrails", upload=upload, use_judge=False,
                     prefix="guardrails", metadata={"suite": "guardrails"}, concurrency=4)


def _print_rows(label: str, rows) -> int:
    passed = sum(r["passed"] for r in rows)
    print(f"{'PASS' if passed == len(rows) else 'FAIL'}  {label:38} {passed}/{len(rows)}")
    for r in rows:
        if not r["passed"]:
            print(f"      ✗ inputs={json.dumps(r['inputs'], default=str)[:110]}")
            for key, score, comment in r["checks"]:
                if not score:
                    print(f"          {key}: {(comment or '')[:160]}")
    return len(rows) - passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", choices=["langgraph", "langchain", "both"], default="both")
    parser.add_argument("--agents", help="comma-separated agent numbers, e.g. 01,03,11 (default: all)")
    parser.add_argument("--local", action="store_true", help="don't upload datasets/experiments or trace to LangSmith")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge checks")
    parser.add_argument("--guardrails-only", action="store_true")
    parser.add_argument("--skip-guardrails", action="store_true")
    args = parser.parse_args()

    upload = not args.local and bool(os.getenv("LANGCHAIN_API_KEY"))
    if not upload:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # set before agent modules load
    from cases import SUITES

    frameworks = {"langgraph": ["LangGraph"], "langchain": ["LangChain"], "both": ["LangGraph", "LangChain"]}[args.framework]
    agents = args.agents.split(",") if args.agents else sorted(SUITES)
    print(f"Evals → {'LangSmith experiments + local summary' if upload else 'local only'}; "
          f"judge {'off' if args.no_judge else 'on'}\n")

    failures = 0
    if not args.skip_guardrails:
        failures += _print_rows("guardrails", run_guardrail_evals(upload=upload))
    if not args.guardrails_only:
        for fw in frameworks:
            for num in agents:
                try:
                    rows = run_agent_evals(fw, num, upload=upload, use_judge=not args.no_judge)
                except Exception as exc:
                    print(f"ERROR {fw}/{num}: {type(exc).__name__}: {exc}")
                    failures += 1
                    continue
                failures += _print_rows(f"{fw} {num}", rows)

    print(f"\n{'All evals passed' if not failures else f'{failures} failing example(s)'}")
    if upload:
        print("Experiments: https://smith.langchain.com → Datasets & Experiments → aiagents-*")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
