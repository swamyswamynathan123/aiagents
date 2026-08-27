# CLAUDE.md — AI Agents Project

## What this project is
16 standalone Python modules demonstrating LangGraph agent patterns.
Each file is independently runnable with `python <filename>.py`.

## Stack
- LangGraph >= 1.2.0 — graph orchestration
- LangChain >= 0.3.0 — LLM abstractions
- LangSmith >= 0.2.0 — tracing (auto-enabled via env vars)
- OpenAI gpt-4o-mini — LLM backend for all modules
- FAISS — in-memory vector store (module 14 only)

## Environment
- Python 3.11+, venv at `./venv/`
- Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
- Required env vars in `.env`: `OPENAI_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`

## Module naming convention
`##_<agent_type>.py` — numbered for learning progression, 01 (simplest) → 16 (most complex)

## Key API notes
- `create_react_agent` from `langgraph.prebuilt` is deprecated in 1.2.x (module 02 keeps it for reference)
- `interrupt()` from `langgraph.types` is the correct HITL API (not compile-time `interrupt_before` flags)
- `Send()` from `langgraph.types` drives parallel fan-out (module 13)
- `MemorySaver` is dev-only — use `PostgresSaver` in production
- All modules use `load_dotenv()` and `gpt-4o-mini` by default

## Running
```bash
python 01_simple_tool_agent.py    # simplest — tool loop
python 07_human_in_loop_agent.py  # two-step: first invoke pauses, second resumes
python 14_rag_agent.py            # requires faiss-cpu installed
```

## LangSmith traces
Set `LANGCHAIN_TRACING_V2=true` in `.env` → all runs auto-traced at https://smith.langchain.com
Project name: `aiagents-demo` (set via `LANGCHAIN_PROJECT`)

## Do not
- Do not use `PostgresSaver` in these demos (requires a running Postgres instance)
- Do not add real external tool integrations (all tools are mocked for portability)
- Do not break the single-file self-contained pattern — each module must run standalone
