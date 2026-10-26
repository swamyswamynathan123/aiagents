# Project Plan and Status

## Goal
A learning repository for agent design: 16 self-contained modules showing the major agent
patterns in LangGraph and LangChain, capstone applications that combine them, and the testing,
evaluation and guardrail practices that make agents dependable. The design reasoning lives in
[guide.md](guide.md); how to run everything is in [README.md](README.md).

## Delivered

| Phase | What | Where |
|-------|------|-------|
| 1 | 16 pattern modules (LangGraph) | `01_*.py` … `16_*.py` |
| 2 | Live web search via Tavily instead of mocked data (01, 02, 10) | `langchain-tavily` |
| 3 | Streamlit UI for every pattern | `app.py` |
| 4 | The same 16 patterns in LangChain only (no LangGraph) | `langchain/` |
| 5 | Guardrails: injection, moderation, PII, length | `guardrails.py` |
| 6 | Evals with LangSmith experiments and a red-team suite | `evals/` |
| 7 | Offline unit tests with a scripted fake LLM; CI | `tests/`, `.github/workflows/` |
| 8 | Capstone: customer support assistant | `projects/customer_support/` |
| 9 | Capstone: research report generator | `projects/research_report/` |
| 10 | Capstone: data analyst over CSV | `projects/data_analyst/` |
| 11 | Design guide | `guide.md` |
| 12 | Combined multipage app, localhost only | `main.py`, `.streamlit/config.toml` |
| 13 | Configurable models (`AGENT_MODEL`), line-ending normalization | `.env.example`, `.gitattributes` |
| 14 | Capstone: document Q&A with hybrid retrieval and a retrieval benchmark | `projects/doc_qa/` |

## Pattern coverage

| # | Pattern | Module | Used in capstones |
|---|---------|--------|-------------------|
| 01 | Tool loop | simple_tool_agent | support, data analyst |
| 02 | ReAct / web tools | react_agent | research |
| 03 | Router | router_agent | support, document Q&A |
| 04 | Plan-and-execute | planning_agent | research |
| 05 | Reflection | reflection_agent | data analyst, document Q&A |
| 06 | Memory | memory_agent | support, document Q&A |
| 07 | Human-in-the-loop | human_in_loop_agent | support, research |
| 08 | Supervisor | multi_agent_supervisor | research |
| 09 | Handoff | handoff_agent | support |
| 10 | Autonomous agent | autonomous_agent | data analyst |
| 11 | Structured output | structured_output_agent | data analyst, document Q&A |
| 12 | Code execution | code_execution_agent | data analyst |
| 13 | Parallel fan-out | parallel_agent | research |
| 14 | RAG | rag_agent | support, document Q&A |
| 15 | Evaluator-optimizer | evaluator_optimizer_agent | research |
| 16 | Chatbot | chatbot_agent | support |

## Roadmap (not started)
- Scheduled real-API evals with pass-rate tracking; repeated runs for flaky cases
- Indirect-injection filtering on tool output in pattern modules 01, 02 and 10
- Persistent state (`SqliteSaver`) so paused approvals and conversations survive restarts
- Per-session spending caps
- Container sandbox for model-written code
- Replace deprecated `RunnableWithMessageHistory` in `langchain/06` and `16`; bump GitHub Actions versions
- `start.sh` for macOS/Linux; Dependabot with CI-gated upgrades
- Further capstones: a coding agent that fixes failing tests; a durable ops incident responder

## Constraints (see CLAUDE.md)
- Pattern modules stay single-file and runnable on their own; capstones are multi-file packages
- External services: OpenAI and Tavily only; other tools are mocked
- No `PostgresSaver` (needs a running database)
