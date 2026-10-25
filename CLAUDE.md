# CLAUDE.md — AI Agents Project

## What this project is
16 standalone Python modules demonstrating LangGraph agent patterns, plus the same 16
patterns re-implemented with LangChain only (no LangGraph) in `langchain/`.
Each file is independently runnable with `python <filename>.py`. `app.py` is a Streamlit UI for both sets,
`guardrails.py` adds input/output safety checks, and `evals/` scores every agent (LangSmith experiments).

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
start.bat                         # Windows: sets up venv/.env, launches Streamlit UI (app.py)
streamlit run app.py              # UI for all 16 agents
python evals/run_evals.py         # all evals, both frameworks (~10 min)
python guardrails.py              # guardrails demo
python 01_simple_tool_agent.py    # simplest — tool loop
python langchain/01_simple_tool_agent.py  # same agent, LangChain-only
python 07_human_in_loop_agent.py  # two-step: first invoke pauses, second resumes
python 14_rag_agent.py            # requires faiss-cpu installed
```

## LangSmith traces
Set `LANGCHAIN_TRACING_V2=true` in `.env` → all runs auto-traced at https://smith.langchain.com
Project name: `aiagents-demo` (set via `LANGCHAIN_PROJECT`)

## Frontend (`app.py`)
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

## Evals (`evals/`)
- `evals/cases.py`: `SUITES[NN]` = `call(ctx, inputs)` that normalizes output to `{answer, tools, fields}` + examples whose `reference` declares checks (`must_contain`, `must_use_tools`, `fields`, `min_fields`/`max_fields`, `criteria` for the LLM judge). `GUARDRAIL_SUITE` is the red-team set
- `python evals/run_evals.py [--agents 03,11] [--framework langchain] [--local] [--no-judge] [--guardrails-only]`; exits 1 on any failure
- With `LANGCHAIN_API_KEY`, suites sync to LangSmith datasets `aiagents-<suite>-<hash>` (new hash when cases change) and runs are experiments prefixed `<framework>-<NN>`; `--local` uses `evaluate(upload_results=False)` and disables tracing
- Adding an agent means adding a `SUITES` entry; keep cases cheap (1-3 per agent) and assertions robust to LLM wording

## Do not
- Do not use `PostgresSaver` in these demos (requires a running Postgres instance)
- Do not add real external tool integrations beyond Tavily search (other tools stay local/mocked for portability)
- Do not break the single-file self-contained pattern — each module must run standalone
