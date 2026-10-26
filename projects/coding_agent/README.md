# Coding Agent

A LangGraph capstone that fixes failing tests in a small Python repo. It reads and searches the
code, edits files and runs tests, and returns a diff. Here the **test suite is the verifier**: no LLM
judges whether the fix worked.
The full design write-up is in [guide.md](../../guide.md#13-capstone-coding-agent).

| Pattern | Where |
|---------|-------|
| Tool loop (01) / autonomous agent (10) | `agent` ⇄ `executor` with `list_files`, `read_file`, `search_code`, `run_tests`, `edit_file`, `finish` |
| Planning (04) | The agent must state a root-cause hypothesis before editing |
| Code execution (12) | Tests run in a fresh `python -I` subprocess with no API keys and a timeout |
| Reflection (05) | `verify` reruns the full suite after `finish`; failures go back once |
| Guardrails | Paths confined to the repo, **test files read-only**, risky edits rejected, **regressing edits reverted** |

```bash
python -m projects.coding_agent.demo [task-id ...]   # e.g. tier-boundary two-bugs
streamlit run projects/coding_agent/ui.py            # pick a task, watch it work, review the diff
python -m projects.coding_agent.evals                # 7 tasks, graded with hidden tests
pytest tests/test_coding_agent.py                    # 26 offline tests (real pytest runs)
AGENT_MODEL=gpt-4o python -m projects.coding_agent.demo two-bugs   # compare models
```

**The sample repo** (`sample_repo/`) is a small shop library: pricing with bulk discounts, a cart
and inventory reservations, with 14 tests. Each task in `tasks.py` injects one known bug:
- a tier boundary that's off by one;
- a percentage discount divided by 10 instead of 100;
- removing the exact quantity leaving an empty line;
- half-even instead of half-up rounding;
- reservations being ignored, which allows overselling;
- a missing `clear()` method;
- two of these bugs together.

**Grading** (`evals.py`):
- the visible tests pass;
- **hidden tests** the agent never sees also pass (`hidden_tests/`), which catches fixes that only special-case the visible tests;
- no test file was changed;
- the diff stays small.

| File | Role |
|------|------|
| `graph.py` | Graph, rules in the executor (rollback, budgets, thrash detection), `CodingSession` |
| `workspace.py` | Tool implementations: path rules, exact-match edits, isolated test runner, diffs |
| `tasks.py` | Task definitions (bug mutations), workspace creation, hidden-test grading copy |
| `sample_repo/`, `hidden_tests/` | The clean repo and the tests the agent never sees |
| `demo.py` / `ui.py` / `evals.py` | CLI demo, Streamlit UI, task evals |

The isolation is demo-grade, like the data analyst's sandbox: run model-written code from
untrusted users in a container.
