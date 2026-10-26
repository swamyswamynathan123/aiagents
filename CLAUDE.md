# CLAUDE.md — AI Agents Project

## What this project is
16 standalone Python modules demonstrating LangGraph agent patterns, plus the same 16
patterns re-implemented with LangChain only (no LangGraph) in `langchain/`.
Each file is independently runnable with `python <filename>.py`. `app.py` is a Streamlit UI for both sets,
`guardrails.py` adds input/output safety checks, and `evals/` scores every agent (LangSmith experiments).
`projects/customer_support/`, `projects/research_report/` and `projects/data_analyst/` are multi-file LangGraph capstones combining several patterns.
`guide.md` explains every pattern and the design decisions; update it when patterns or lessons change.

## Stack
- LangGraph >= 1.2.0 — graph orchestration
- LangChain >= 0.3.0 — LLM abstractions
- LangSmith >= 0.2.0 — tracing (auto-enabled via env vars)
- OpenAI gpt-4o-mini — LLM backend for all modules
- Tavily (`langchain-tavily`) — live web search backing the tools in modules 01, 02, 10
- FAISS — in-memory vector store (module 14 only)

## Environment
- Python 3.11+, venv at `./venv/`
- Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
- Required env vars in `.env`: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`

## Module naming convention
`##_<agent_type>.py` — numbered for learning progression, 01 (simplest) → 16 (most complex)

## Key API notes
- `create_react_agent` from `langgraph.prebuilt` is deprecated in 1.2.x (module 02 keeps it for reference)
- `interrupt()` from `langgraph.types` is the correct HITL API (not compile-time `interrupt_before` flags)
- `Send()` from `langgraph.types` drives parallel fan-out (module 13)
- `MemorySaver` is dev-only — use `PostgresSaver` in production
- All modules use `load_dotenv()` and `gpt-4o-mini` by default
- Tavily: `TavilySearch` from `langchain_tavily` is wrapped in `@tool` functions; each module carries its own `_format_results` helper (kept per-file to stay self-contained). Use `topic="news"`/`"finance"` for news/stock lookups. The client is built at import time, so a missing `TAVILY_API_KEY` fails on startup

## Running
```bash
start.bat                         # Windows: sets up venv/.env, launches the multipage app (main.py)
streamlit run main.py             # one app: home, 16 patterns, 3 capstones (localhost only)
pytest                            # offline unit tests (~25s, no API keys)
python evals/run_evals.py         # all evals, both frameworks (~10 min)
python guardrails.py              # guardrails demo
python -m projects.customer_support.demo       # capstone demo
streamlit run projects/customer_support/ui.py  # capstone UI (chat + supervisor console)
streamlit run projects/research_report/ui.py   # research capstone UI (plan review, live progress)
streamlit run projects/data_analyst/ui.py      # data analyst UI (CSV upload, charts, show work)
python 01_simple_tool_agent.py    # simplest — tool loop
python langchain/01_simple_tool_agent.py  # same agent, LangChain-only
python 07_human_in_loop_agent.py  # two-step: first invoke pauses, second resumes
python 14_rag_agent.py            # requires faiss-cpu installed
```

## LangSmith traces
Set `LANGCHAIN_TRACING_V2=true` in `.env` → all runs auto-traced at https://smith.langchain.com
Project name: `aiagents-demo` (set via `LANGCHAIN_PROJECT`)

## Frontend (`main.py` + pages)
- `main.py` is a `st.navigation` multipage app: Home, `app.py` (patterns), and each capstone's `ui.py`. Every page still runs standalone
- Pages share `st.session_state`: prefix keys per page (`support_`, `report_`, `analyst_`; `app.py` uses `<framework>_<NN>_`). `tests/test_main_app.py` checks pages don't overwrite each other
- `.streamlit/config.toml` binds the server to localhost; keep it that way (the agents spend API credit and run model-written code)

### Patterns page (`app.py`)
- Streamlit UI that imports each module via `importlib` (cached with `st.cache_resource`) and calls its compiled `app` — modules are never modified for the UI
- Sidebar switches framework (LangGraph root vs `langchain/`). Each `page_NN(ctx)` renders inputs/results and calls `ctx["run"]`: the LangChain module's `run()`, or a `LANGGRAPH_RUN[NN]` adapter in `registry.py` that maps inputs onto the graph state
- Tabs: Run / Built-in demo (`main()` output) / Evals (this agent's cases) / Source
- Adding a module means adding its page function, a LangGraph adapter, a LangChain `run()`, and an eval suite
- "Built-in demo" tab runs the module's `main()` with stdout captured

## LangChain versions (`langchain/`)
- Same filenames, prompts and demos as the root modules. Built from LCEL (`prompt | llm | parser`), `bind_tools`, `with_structured_output`, `RunnableWithMessageHistory`, `.batch()`, with control flow in plain Python
- Must not import `langgraph` (including `langchain.agents.create_agent`, which is LangGraph-backed)
- UI contract: each exposes `run(**inputs) -> dict` returning the same keys the LangGraph final state has (06/16: `chat(...)`; 07: `propose_action(task)` + `resolve(action, approved)`)
- No `__init__.py` in `langchain/`: it would shadow the installed `langchain` package when running from the project root

## Shared support modules
- `registry.py`: module loading + uniform calling interface (`make_runner`, `hitl_api`, LangGraph state adapters). Used by `app.py` and `evals/`; agents never import it
- `guardrails.py`: `Guard(GuardrailConfig(...))` with `check_input` (length, prompt-injection regex + optional LLM classifier, OpenAI moderation, PII redaction) and `check_output` (PII redaction + moderation). `guard.wrap(run)` guards str/list[str] kwargs and the whole result; blocks raise `GuardrailBlocked`. Moderation fails open (warns) if the API errors
- The UI applies guardrails around agents (sidebar toggles); `PII_EXEMPT` in `app.py` turns off redaction for agents that need PII (11)

## Tests (`tests/`)
- `pytest` runs offline in ~25s: `conftest.py` sets dummy keys, disables tracing, installs `tests/fakes.py` (scripted `FakeChat` BaseChatModel + fake embeddings/Tavily) before agent modules load, and stubs `guardrails.moderate`
- Steer the fake per test via `fakes.script`: `structured[SchemaName]`, `reply(system, human)`, `tool_calls`, `tool_order`; inspect `script.prompts` / `script.calls`
- Agent tests take the `framework` fixture so every behavior is asserted for both LangGraph and LangChain; `load()` gives a fresh module per test
- Any loop an LLM can keep going (tool calls, replans, handoffs) must have an explicit cap with a test — LangGraph 1.2's default recursion_limit is ~10k
- CI: `.github/workflows/tests.yml` (every push, no secrets); `evals.yml` is manual-only (needs OPENAI/TAVILY/LANGCHAIN secrets)
- Dependencies are pinned in `requirements.txt`; to upgrade, bump → `pytest` → `python evals/run_evals.py`. Dev deps: `requirements-dev.txt`

## Evals (`evals/`)
- `evals/cases.py`: `SUITES[NN]` = `call(ctx, inputs)` that normalizes output to `{answer, tools, fields}` + examples whose `reference` declares checks (`must_contain`, `must_use_tools`, `fields`, `min_fields`/`max_fields`, `criteria` for the LLM judge). `GUARDRAIL_SUITE` is the red-team set
- `python evals/run_evals.py [--agents 03,11] [--framework langchain] [--local] [--no-judge] [--guardrails-only]`; exits 1 on any failure
- With `LANGCHAIN_API_KEY`, suites sync to LangSmith datasets `aiagents-<suite>-<hash>` (new hash when cases change) and runs are experiments prefixed `<framework>-<NN>`; `--local` uses `evaluate(upload_results=False)` and disables tracing
- Adding an agent means adding a `SUITES` entry; keep cases cheap (1-3 per agent) and assertions robust to LLM wording

## Capstone projects (`projects/`)
- Multi-file packages, LangGraph only; run from the repo root as modules (`python -m projects.customer_support.demo`)
- `customer_support/`: `graph.py` (nodes + `SupportSession` API), `tools.py` (`customer_id` is an `InjectedToolArg` — never let the model choose it), `policies.py` (business rules as code), `data.py` (mock backend, `reset()`), `kb.py` (FAISS help center), `ui.py`, `evals.py` (11 scenarios, reuses `evals/run_evals.run_suite`)
- Side effects in an interrupted node go after `interrupt()` (the node re-runs on resume); re-check policy after approval
- Tests in `tests/test_support.py`; reset the backend per test and use `build_app()` for an isolated graph
- `research_report/`: `graph.py` (planner → plan_review interrupt → `Send` fan-out to `research` → writer ⇄ reviewer → finalize; `ReportSession` streams updates and tracks token usage), `search.py` (Tavily + drops results with injection patterns). Citations: code numbers sources, validates `[n]`, writes References — the model never writes URLs. Budgets: `MAX_QUESTIONS`, `MAX_RESEARCH_ROUNDS`, `MAX_REVISIONS`. Tests in `tests/test_research.py` (`FakeTavily.respond` scripts search results)
- `data_analyst/`: `graph.py` (profile → understand `AnalysisSpec` → analyst ⇄ executor → verify → finalize; `AnalystSession`), `sandbox.py` (AST import allow-list, fresh `python -I` subprocess with stripped env, notebook-style replay of earlier ok cells + last-expression display, `save_chart()`), `verify.py` (every number in the answer must appear in an execution output). Sample data from `make_data.py` (deterministic). Tests in `tests/test_data_analyst.py` run real pandas in the sandbox

## Do not
- Do not use `PostgresSaver` in these demos (requires a running Postgres instance)
- Do not add real external tool integrations beyond Tavily search (other tools stay local/mocked for portability)
- Do not break the single-file self-contained pattern for the 16 pattern modules (capstones in `projects/` are multi-file by design) — each module must run standalone
