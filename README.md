# LangGraph + LangSmith AI Agents

16 self-contained Python modules demonstrating every major LangGraph agent pattern, with LangSmith observability and OpenAI as the LLM backend.

---

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
setup_venv.bat          # Windows
bash setup_venv.sh      # macOS / Linux

# 2. Configure API keys
copy .env.example .env  # Windows
cp .env.example .env    # macOS / Linux
# Edit .env and fill in OPENAI_API_KEY, TAVILY_API_KEY and LANGCHAIN_API_KEY
# (free Tavily key: https://app.tavily.com — used by modules 01, 02, 10)

# 3. Run any agent
python 01_simple_tool_agent.py
python 16_chatbot_agent.py
```

---

## Stack

| Component | Package | Purpose |
|-----------|---------|---------|
| LangGraph | `langgraph>=1.2.0` | Graph orchestration, state management, agent loops |
| LangChain | `langchain>=0.3.0` | LLM abstractions, tools, prompt templates |
| LangSmith | `langsmith>=0.2.0` | Tracing, debugging, evaluation |
| OpenAI | `langchain-openai>=0.2.0` | LLM backend (`gpt-4o-mini`) |
| Tavily | `langchain-tavily>=0.2.0` | Live web search for tools (modules 01, 02, 10) |
| FAISS | `faiss-cpu>=1.7.0` | In-memory vector store (module 14 only) |
| Pydantic v2 | `pydantic>=2.0.0` | Type-safe state and structured output |

---

## Agent Modules

### 01 — Simple Tool Agent
**File:** [01_simple_tool_agent.py](01_simple_tool_agent.py)  
**Pattern:** Manual `StateGraph` with a hand-written tool-call loop. No prebuilt helpers.

**How it works:**
1. `call_llm` node invokes the LLM with tools bound
2. `should_continue` routing checks for tool calls in the last message
3. `execute_tools` node runs each tool and returns `ToolMessage` results
4. Loop repeats until the LLM produces a final answer with no tool calls

**Key LangGraph concepts:** `StateGraph`, `add_messages` reducer, `add_conditional_edges`

**Use cases:** Learning LangGraph mechanics, custom tool execution logic, fine-grained control over the tool loop

**When to use:** When you need full control over how tools are executed and results are formatted, or when `create_react_agent` doesn't fit your requirements.

---

### 02 — ReAct Agent
**File:** [02_react_agent.py](02_react_agent.py)  
**Pattern:** Reasoning + Acting loop using `create_react_agent` from `langgraph.prebuilt`.

> **Deprecation note:** `create_react_agent` from `langgraph.prebuilt` is deprecated in LangGraph 1.2.x. The successor is `create_agent` from `langchain>=0.3`. This module is kept for reference.

**How it works:**
- The prebuilt helper wires an `agent` node and a `tools` node with automatic routing
- The LLM reasons about which tool to use (Reason), executes it (Act), observes the result, then repeats

**Key LangGraph concepts:** `create_react_agent`, tool binding, prebuilt graph

**Use cases:** General-purpose tool-using assistants, research bots, multi-step Q&A requiring external lookups

**When to use:** Fast prototyping when you need a tool-using agent without custom control flow.

---

### 03 — Router Agent
**File:** [03_router_agent.py](03_router_agent.py)  
**Pattern:** A router node classifies the query and routes via `add_conditional_edges` to specialized sub-nodes.

**How it works:**
1. `router_node` calls the LLM with `with_structured_output(RouteDecision)` to classify the query
2. `route_selector` reads `state["route"]` and returns the destination node name
3. Specialized nodes (`math`, `writing`, `general`) handle the query with domain-specific system prompts

**Key LangGraph concepts:** `add_conditional_edges`, `Literal` type routing, `with_structured_output`

**Use cases:** Intent classification, multi-department chatbots, routing queries to specialist models, domain-specific responses

**When to use:** When different query types need fundamentally different handling logic or system prompts.

---

### 04 — Planning Agent
**File:** [04_planning_agent.py](04_planning_agent.py)  
**Pattern:** Plan-and-Execute — Planner generates a multi-step plan; Executor runs each step; Re-planner adjusts or finalizes.

**How it works:**
1. `planner_node` generates an ordered list of steps (`Plan` Pydantic schema)
2. `executor_node` runs one step at a time, accumulating results
3. `replanner_node` evaluates progress and either produces a final answer or revises the plan
4. Control flows between executor and replanner until the goal is met

**Key LangGraph concepts:** Multi-node pipeline, `with_structured_output`, conditional routing between 3+ nodes

**Use cases:** Complex research tasks, automated project management, multi-step workflows, tasks requiring upfront decomposition before execution

**When to use:** When the full plan must be known before execution begins, or when steps depend on the results of prior steps.

---

### 05 — Reflection Agent
**File:** [05_reflection_agent.py](05_reflection_agent.py)  
**Pattern:** Generator produces a draft; Critic evaluates it; loop continues until quality threshold is met or `max_iterations` is reached.

**How it works:**
1. `generator_node` creates a draft (or revises based on critique feedback on subsequent iterations)
2. `critic_node` scores the draft and decides whether to approve, using `CritiqueResult` schema
3. `reflection_router` ends if approved or iteration limit hit, else routes back to generator
4. `iteration` counter prevents infinite loops

**Key LangGraph concepts:** Cyclic graph, iteration cap, `with_structured_output` for structured critique

**Use cases:** Self-improving content generation, iterative code writing, essay drafting, automated document quality improvement

**When to use:** When quality matters more than speed and you want the agent to self-correct before returning output.

---

### 06 — Memory Agent
**File:** [06_memory_agent.py](06_memory_agent.py)  
**Pattern:** `MemorySaver` (per-thread conversation history) + `InMemoryStore` (cross-thread user facts).

**Dual-layer memory architecture:**
| Layer | Class | Scope | Contents |
|-------|-------|-------|---------|
| Short-term | `MemorySaver` | Per `thread_id` | Full message history |
| Long-term | `InMemoryStore` | Cross-thread (by `user_id`) | User facts, preferences |

**How it works:**
1. `recall_and_respond` node retrieves user facts from the store, builds a personalized system prompt, calls the LLM, and heuristically extracts new facts to save back
2. `MemorySaver` automatically restores the full conversation when the same `thread_id` is used
3. `InMemoryStore` persists facts across different threads for the same user

**Key LangGraph concepts:** `checkpointer`, `store`, `thread_id`, `user_id`, store injection via node parameter

**Production note:** Replace `MemorySaver` with `PostgresSaver` and `InMemoryStore` with `PostgresStore` for durability.

**Use cases:** Personalized assistants, multi-session chatbots, agents that remember user preferences long-term

---

### 07 — Human-in-the-Loop Agent
**File:** [07_human_in_loop_agent.py](07_human_in_loop_agent.py)  
**Pattern:** `interrupt()` pauses graph execution for human review; `Command(resume=value)` continues with the human's decision.

**Two-step invocation:**
```python
# Step 1: run until interrupt
snapshot = app.invoke({"task": "..."}, config)
# Step 2: resume with human decision
final = app.invoke(Command(resume=True), config)  # True = approve, False = reject
```

**How it works:**
1. `plan_action` generates a proposed action
2. `request_approval` calls `interrupt({...})` — graph halts and returns to caller
3. Human reviews the proposed action and resumes with approval/rejection
4. `approval_router` routes to `execute` or `skip` based on the human's decision

**Key LangGraph concepts:** `interrupt()` (from `langgraph.types`), `Command(resume=...)`, `MemorySaver` (required for interrupt state persistence)

**Use cases:** Email/message sending approval, database delete confirmation, code deployment gates, financial transaction authorization, any sensitive operation requiring human oversight

---

### 08 — Multi-Agent Supervisor
**File:** [08_multi_agent_supervisor.py](08_multi_agent_supervisor.py)  
**Pattern:** Central Supervisor decides which worker to delegate to; workers complete tasks and return to the Supervisor until the objective is met.

**How it works:**
1. `supervisor_node` uses `with_structured_output(RoutingDecision)` to decide: `researcher`, `writer`, or `FINISH`
2. `researcher_node` gathers information and returns to supervisor
3. `writer_node` drafts content using research results and returns to supervisor
4. Supervisor loops until routing returns `FINISH`

**Key LangGraph concepts:** Supervisor loop pattern, structured routing with Pydantic, iteration guard

**Use cases:** Content pipelines (research → write), complex analyses with multiple expert roles, automated report generation, multi-role customer service workflows

**When to use vs. Handoff (09):** Use Supervisor when a central coordinator should manage flow; use Handoff when agents should self-direct transfers.

---

### 09 — Handoff Agent (Swarm Pattern)
**File:** [09_handoff_agent.py](09_handoff_agent.py)  
**Pattern:** Agents dynamically transfer control using `Command(goto=<agent_name>)`. No central supervisor — agents self-direct based on query type.

**How it works:**
1. Query starts at `triage_agent`
2. Each agent node returns `Command(goto=<target>, update={...})` to hand off, or `Command(goto=END)` to respond
3. Handoff is detected when the LLM returns a `HANDOFF:<agent>` marker in its response
4. State tracks `active_agent` and `handoff_count`

**Key LangGraph concepts:** `Command(goto=...)` for dynamic routing, node-level routing (not edge-level)

**Implementation note:** This module implements handoff natively with `Command` without requiring the separate `langgraph-swarm` package.

**Use cases:** Customer service routing (triage → billing/technical), tiered support, specialist agent networks, any workflow where domain determines routing

---

### 10 — Autonomous Agent
**File:** [10_autonomous_agent.py](10_autonomous_agent.py)  
**Pattern:** Self-directed, long-running agent that autonomously selects tools and drives toward a goal. `MemorySaver` enables resumable sessions. `max_iterations` prevents runaway loops.

**How it works:**
1. `autonomous_agent_node` receives the goal + progress notes in its system prompt; selects the next action
2. `tool_executor_node` runs all tool calls, extracts notes from `save_note` calls, detects `mark_goal_complete`
3. `autonomous_router` ends when `completed=True` or iteration limit is hit
4. Resumes across invocations via `thread_id` + `MemorySaver`

**Key LangGraph concepts:** Persistent session via `MemorySaver` + `thread_id`, self-directed tool loop, `completed` flag termination

**Use cases:** Background research, automated report generation, long-horizon tasks, AI assistants that work independently without human prompting between steps

---

### 11 — Structured Output Agent
**File:** [11_structured_output_agent.py](11_structured_output_agent.py)  
**Pattern:** Single-node extraction using `llm.with_structured_output()` + Pydantic v2. One LLM call → validated, type-safe structured data.

**Key schemas:**
- `ContactInfo` — name, email, phone, company, role
- `MeetingRequest` — date, time, duration, participants, topic
- `EmailAnalysis` — nested contact + meeting + sentiment + priority + action items + summary

**How it works:**
1. `extract_node` calls `_extractor.invoke([...])` where `_extractor = llm.with_structured_output(EmailAnalysis)`
2. The LLM returns JSON that Pydantic v2 validates into a typed `EmailAnalysis` object
3. Node stores `result.model_dump()` in state

**Key LangGraph concepts:** `with_structured_output`, Pydantic v2 `Field(description=...)`, nested schemas

**Use cases:** Data extraction from unstructured text, form field population, entity recognition, document parsing, ETL pipelines from natural language

---

### 12 — Code Execution Agent
**File:** [12_code_execution_agent.py](12_code_execution_agent.py)  
**Pattern:** LLM generates Python code → sandboxed `subprocess` executes it → output feeds back to LLM for interpretation.

**Safety measures:**
- Blocked patterns: `import os`, `import sys`, `import subprocess`, `open(`, `exec(`, `__import__`, and others
- 10-second execution timeout
- Runs in a temp directory

**How it works:**
1. `code_agent_node` prompts the LLM to write pure computational Python
2. LLM calls `execute_python` tool with code string
3. `tool_exec_node` applies safety checks, writes code to a temp file, runs via `subprocess`
4. Output returns to LLM for explanation or debugging

**Key LangGraph concepts:** Tool with side effects, safety filtering, execution count cap

**Use cases:** Data analysis assistants, automated computation, REPL-style coding environments, math problem solving

---

### 13 — Parallel Agent (Fan-Out / Map-Reduce)
**File:** [13_parallel_agent.py](13_parallel_agent.py)  
**Pattern:** `Send()` from `langgraph.types` dispatches multiple independent tasks concurrently. `operator.add` reducer aggregates all parallel results.

**How it works:**
1. `fan_out` function returns `List[Send("analyze_item", {item_state})]` — one per item
2. LangGraph dispatches all `analyze_item` nodes concurrently
3. Each returns `{"analyses": ["..."]}` — `operator.add` reducer appends all results
4. After all parallel nodes complete, `aggregate_results` synthesizes a summary

**Key LangGraph concepts:** `Send()`, `operator.add` as Annotated reducer, `START` → fan-out, fan-in via edge to shared node

**Use cases:** Batch document analysis, parallel evaluation, concurrent web scraping, any workflow where items are independent and can be processed simultaneously

---

### 14 — RAG Agent (Retrieval-Augmented Generation)
**File:** [14_rag_agent.py](14_rag_agent.py)  
**Pattern:** Retrieve semantically similar documents from a FAISS vector store → inject as context → generate grounded answer.

**How it works:**
1. `retrieve_node` calls `vectorstore.similarity_search(query, k=N)` to find the most relevant docs
2. `generate_node` injects retrieved docs into the system prompt as numbered context
3. LLM is instructed to answer ONLY using the provided context

**Stack:** `FAISS.from_texts()` + `OpenAIEmbeddings(model="text-embedding-3-small")`

**Key LangGraph concepts:** Two-node pipeline, external tool/library in node function, lazy initialization

**Production upgrades:** Replace `FAISS` with Pinecone/Weaviate/pgvector; replace `KNOWLEDGE_BASE` list with a document loader pipeline.

**Use cases:** Document Q&A, internal knowledge bases, customer support with product documentation, legal/medical document analysis

---

### 15 — Evaluator-Optimizer Agent
**File:** [15_evaluator_optimizer_agent.py](15_evaluator_optimizer_agent.py)  
**Pattern:** Generator produces output → separate Evaluator LLM scores against a rubric → loop until target score reached.

**Difference from Reflection (05):**
| | Reflection | Evaluator-Optimizer |
|--|--|--|
| Critic | Same model, self-critique | Separate LLM call with explicit rubric |
| Score | Qualitative (approved/not) | Quantitative (1–10) with threshold |
| Focus | General quality | Measurable, rubric-defined criteria |

**Rubric dimensions (each 0–2 points, total 10):**
1. Clarity
2. Benefit focus
3. Specificity
4. Emotional appeal
5. Call to action

**Key LangGraph concepts:** Score-based loop termination, separate generator and evaluator LLM temperature settings, `EvalResult` structured schema with `ge`/`le` validators

**Use cases:** Marketing copy optimization, legal/compliance document review, prompt A/B testing, API documentation quality gates

---

### 16 — Chatbot Agent
**File:** [16_chatbot_agent.py](16_chatbot_agent.py)  
**Pattern:** Minimal multi-turn conversational agent. One node, one edge, `MemorySaver` for history. The foundation all other patterns build on.

**How it works:**
1. `chat_node` prepends a system prompt and calls the LLM with the full message history
2. `add_messages` reducer appends the new AI response to the list
3. `MemorySaver` + `thread_id` persists history across multiple `invoke()` calls

**Key LangGraph concepts:** `add_messages`, `MemorySaver`, `thread_id` configuration

**Interactive mode:** Uncomment the `while True` block in `main()` for live chat.

**Use cases:** Customer service bots, FAQ assistants, interview prep tools, onboarding flows, any application needing stateful conversation

**Start here if you're new to LangGraph** — this is the simplest complete implementation.

---

## LangSmith Observability

All modules auto-trace when these env vars are set:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=aiagents-demo
```

**What you see in LangSmith:**
- Every node execution as a span
- LLM calls with input/output tokens and latency
- Tool calls and their results
- Full message history per thread
- Graph-level execution timeline

View traces at [https://smith.langchain.com](https://smith.langchain.com) under the `aiagents-demo` project.

---

## API Notes

### Key deprecations in LangGraph 1.2.x
| Deprecated | Replacement |
|-----------|-------------|
| `create_react_agent` from `langgraph.prebuilt` | `create_agent` from `langchain>=0.3` |

### Important imports
```python
from langgraph.types import interrupt, Command, Send   # HITL, routing, fan-out
from langgraph.checkpoint.memory import MemorySaver    # dev checkpointer
from langgraph.store.memory import InMemoryStore       # dev store
from langgraph.graph.message import add_messages       # message list reducer
from langgraph.graph import START, END, StateGraph     # graph building blocks
```

### Production notes
- Replace `MemorySaver` → `PostgresSaver` (requires `langgraph-checkpoint-postgres`)
- Replace `InMemoryStore` → `PostgresStore` for durable cross-thread memory
- `thread_id` must be under 255 characters with `PostgresSaver`
- Set `temperature=0` on evaluator/router LLMs for deterministic routing

---

## Project Structure

```
aiagents/
├── CLAUDE.md                       ← Project guide for Claude Code
├── plan.md                         ← Implementation plan
├── requirements.txt
├── .env.example
├── setup_venv.bat / setup_venv.sh
├── 01_simple_tool_agent.py         ← Manual tool loop (foundation)
├── 02_react_agent.py               ← ReAct prebuilt
├── 03_router_agent.py              ← Conditional routing
├── 04_planning_agent.py            ← Plan-and-Execute
├── 05_reflection_agent.py          ← Generator + Critic loop
├── 06_memory_agent.py              ← Short + long-term memory
├── 07_human_in_loop_agent.py       ← interrupt() / approval gate
├── 08_multi_agent_supervisor.py    ← Central supervisor
├── 09_handoff_agent.py             ← Peer-to-peer agent handoff
├── 10_autonomous_agent.py          ← Long-running self-directed
├── 11_structured_output_agent.py   ← Pydantic extraction
├── 12_code_execution_agent.py      ← LLM writes + runs code
├── 13_parallel_agent.py            ← Send() fan-out map-reduce
├── 14_rag_agent.py                 ← Retrieval-Augmented Generation
├── 15_evaluator_optimizer_agent.py ← Generate → Judge → Refine
└── 16_chatbot_agent.py             ← Basic chatbot (start here)
```

---

## Learning Path

**New to LangGraph?** Follow this order:
1. `16_chatbot_agent.py` — minimal graph, message state, MemorySaver
2. `01_simple_tool_agent.py` — add tools and conditional routing
3. `03_router_agent.py` — structured routing with Pydantic
4. `02_react_agent.py` — same as 01 but with prebuilt helper
5. `06_memory_agent.py` — dual-layer memory
6. `07_human_in_loop_agent.py` — interrupt/resume pattern
7. `05_reflection_agent.py` → `15_evaluator_optimizer_agent.py` — quality loops
8. `04_planning_agent.py` — multi-step pipelines
9. `08_multi_agent_supervisor.py` → `09_handoff_agent.py` — multi-agent
10. `13_parallel_agent.py` — fan-out with Send()
11. `14_rag_agent.py` — retrieval integration
12. `10_autonomous_agent.py` — long-running agents
13. `12_code_execution_agent.py` — code-in-the-loop
14. `11_structured_output_agent.py` — extraction pipelines
