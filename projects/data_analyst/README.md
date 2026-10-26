# Data Analyst

A LangGraph capstone that answers questions about a CSV by writing and running real pandas code
in a sandbox, then checks its own answer before showing it.
The full design write-up is in [guide.md](../../guide.md#11-capstone-data-analyst).

| Pattern | Where |
|---------|-------|
| Structured output (11) | `understand` turns the question into an `AnalysisSpec`; code drops column names that don't exist and stops early if the data can't answer the question |
| Code execution (12) | `sandbox.py` runs each script in a fresh, isolated subprocess (allow-listed imports, no API keys in the environment, timeout) |
| Autonomous loop (10) | `analyst` ⇄ `executor`: write code → run → read output → repeat, until `final_answer` (capped) |
| Reflection (05) | `verify`: **every number in the answer must appear in a code output**, plus an LLM review of the method; one revision |
| Guardrails | Question input guard, suspicious CSV cell values withheld from the prompt, output guard |

```bash
python -m projects.data_analyst.demo "your question"   # sample dataset, terminal progress
streamlit run projects/data_analyst/ui.py              # upload a CSV, answers + charts + "show work"
python -m projects.data_analyst.evals                  # 9 scenario evals with pandas-computed ground truth
pytest tests/test_data_analyst.py                      # 51 offline tests (runs real pandas in the sandbox)
```

The sample data (`data/sales_2025.csv`) has 1,500 orders across 4 regions, 3 categories, 9
products and 2 channels, with seasonality and discounts. It's generated deterministically by
`make_data.py`.

| File | Role |
|------|------|
| `graph.py` | State, schemas, prompts, nodes, graph wiring, `AnalystSession` |
| `sandbox.py` | AST allow-list check + isolated subprocess runner + `save_chart()` helper |
| `verify.py` | Number grounding: extracts figures (`$1.2M`, `28.8%`, `744,832`) and matches them to outputs |
| `make_data.py` | Generates the sample dataset |
| `demo.py` / `ui.py` / `evals.py` | CLI demo, Streamlit UI, scenario evals |

The sandbox is stricter than module 12's, but it's still **not** a security boundary for
untrusted users. See guide.md §7.12.
