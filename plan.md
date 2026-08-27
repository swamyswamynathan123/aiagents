# LangGraph AI Agents — Implementation Plan

## Goal
16 self-contained Python modules demonstrating every major LangGraph agent pattern,
with LangSmith tracing and OpenAI as the LLM backend.

## Agent Inventory

| # | Module | Pattern | Use Cases |
|---|--------|---------|-----------|
| 01 | simple_tool_agent | Manual StateGraph tool loop | Learning LangGraph mechanics |
| 02 | react_agent | ReAct prebuilt | General-purpose tool-using assistant |
| 03 | router_agent | Conditional edge routing | Query classification, intent routing |
| 04 | planning_agent | Plan-and-Execute | Multi-step task decomposition |
| 05 | reflection_agent | Generator + Critic loop | Self-improving content generation |
| 06 | memory_agent | MemorySaver + InMemoryStore | Persistent multi-turn conversations |
| 07 | human_in_loop_agent | interrupt() / Command(resume) | Approval gates, sensitive operations |
| 08 | multi_agent_supervisor | Supervisor pattern | Complex multi-role workflows |
| 09 | handoff_agent | Agent-to-agent handoff | Customer service routing |
| 10 | autonomous_agent | Long-running self-directed | Background research, report generation |
| 11 | structured_output_agent | with_structured_output() | Data extraction from unstructured text |
| 12 | code_execution_agent | LLM writes + runs code | Data analysis, scripting assistants |
| 13 | parallel_agent | Send() fan-out map-reduce | Batch processing, parallel analysis |
| 14 | rag_agent | Retrieval-Augmented Generation | Document Q&A, knowledge bases |
| 15 | evaluator_optimizer_agent | Generate → Judge → Refine | Marketing copy, legal/technical writing |
| 16 | chatbot_agent | Basic multi-turn conversation | Customer bots, FAQ assistants |

## Setup
1. Run `setup_venv.bat` (Windows) or `bash setup_venv.sh` (Unix)
2. Copy `.env.example` → `.env`, fill in API keys
3. Run any module: `python 01_simple_tool_agent.py`

## Stack
- LangGraph >= 1.2.0 (graph orchestration)
- LangChain >= 0.3.0 (LLM abstractions)
- LangSmith >= 0.2.0 (tracing / observability)
- OpenAI >= 1.59.0 (LLM backend, gpt-4o-mini)
- FAISS (in-memory vector store for RAG module)

## Key API Notes
- `create_react_agent` from `langgraph.prebuilt` is deprecated in 1.2.x
- `interrupt()` from `langgraph.types` is the stable HITL API
- `Send()` from `langgraph.types` is the fan-out primitive
- `MemorySaver` is for dev only; use `PostgresSaver` in production
- Keep `thread_id` under 255 characters
