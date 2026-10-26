# Research Report Generator

A LangGraph capstone that turns a topic into a cited Markdown report. It plans sub-questions,
researches them in parallel on the web, writes a draft and has it reviewed, within fixed budgets.
The full design write-up is in [guide.md](../../guide.md#10-capstone-research-report-generator).

| Pattern | Where |
|---------|-------|
| Plan-and-execute (04) | `planner` → 3-5 sub-questions with search queries |
| Human-in-the-loop (07) | Optional `plan_review` interrupt: edit or cancel the plan before any research is paid for |
| Parallel fan-out (13) | One `Send("research", …)` per question; results merged by an `operator.add` reducer |
| Web research tools (02) | Tavily search in `search.py` |
| Evaluator-optimizer (15) | `reviewer` scores the draft against a rubric; the `writer` revises |
| Supervisor routing (08) | The reviewer chooses accept, revise or research more (gaps), with code enforcing budgets |
| Guardrails | Topic input guard; **untrusted web pages containing injection text are dropped**; output guard |

```bash
python -m projects.research_report.demo "your topic"   # live progress in the terminal
streamlit run projects/research_report/ui.py           # UI with plan review, report, cost
python -m projects.research_report.evals               # 5 scenario evals (LangSmith experiment)
pytest tests/test_research.py                          # offline tests
```

A typical report uses 10-15k tokens, which costs under $0.01 with gpt-4o-mini.

| File | Role |
|------|------|
| `graph.py` | State, schemas, prompts, nodes, graph wiring, `ReportSession` (streaming, plan review, token usage) |
| `search.py` | Tavily wrapper: truncates results and drops ones containing injection patterns |
| `demo.py` / `ui.py` / `evals.py` | CLI demo, Streamlit UI, scenario evals |

**Citations are handled in code.** Researchers extract facts tied to source numbers. Code assigns
one global, de-duplicated numbering, gives the writer only facts tagged `[n]`, checks every `[n]`
in the draft, and generates the References section itself. The model never writes a URL.
