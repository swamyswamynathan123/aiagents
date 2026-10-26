# Customer Support Assistant

A LangGraph capstone that combines seven of the repo's patterns into one realistic
application: a support assistant for a fictional store, Acme Gadgets.
The full design write-up is in [guide.md](../../guide.md#9-capstone-customer-support-assistant).

| Pattern | Where |
|---------|-------|
| Guardrails | `input_guard` / `output_guard` nodes (injection block, PII redaction, moderation) |
| Router (03) | `triage` classifies every customer message |
| Tool loop (01) | Each specialist calls tools until it can answer (capped at `MAX_STEPS`) |
| Handoff (09) | `transfer_to_agent` moves the conversation between specialists (capped) |
| RAG (14) | `search_help_center` over the help-center articles in `kb.py` |
| Memory (06) | Checkpointer per conversation, plus long-term case notes per customer in the store |
| Human-in-the-loop (07) | Refunds over $50 pause at `interrupt()` until a supervisor decides |

```bash
python -m projects.customer_support.demo         # scripted conversation, real APIs
streamlit run projects/customer_support/ui.py    # chat + supervisor console
python -m projects.customer_support.evals        # 11 scenario evals (LangSmith experiment)
pytest tests/test_support.py                     # offline tests
```

Run these from the repo root. They need the same `.env` as the rest of the repo.

| File | Role |
|------|------|
| `graph.py` | State, nodes, graph wiring, `SupportSession` API |
| `tools.py` | Tools; `customer_id` is injected from the session, never chosen by the model |
| `policies.py` | Refund rules as plain code (30-day window, $50 auto-approve limit) |
| `data.py` | In-memory mock backend: customers, orders, refunds, tickets |
| `kb.py` | Help-center articles + FAISS retrieval |
| `demo.py` / `ui.py` / `evals.py` | CLI demo, Streamlit UI, scenario evals |

Demo customers: **Alice Chen (C001)** has five orders that cover every refund outcome.
**Bob Martin (C002)** owns order B-2001, which Alice must not be able to see.
