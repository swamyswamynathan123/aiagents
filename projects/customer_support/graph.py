"""
Customer support assistant — the LangGraph graph.

Flow for each customer message:

    input_guard ─┬─ blocked ─────────────────────────────────────────────► END
                 └─► triage (every message) ─┬─► billing / technical / general
                                             └─► escalate

    specialist ──► tools ──┬─► same specialist (loop, capped at MAX_STEPS)
        │                  ├─► other specialist (transfer_to_agent, capped at MAX_HANDOFFS)
        │                  └─► approval ── interrupt() until a supervisor decides
        └─ final answer ─► output_guard ──► END

Patterns: guardrails, router (03), tool loop (01), handoff (09), RAG (14),
short- and long-term memory (06), human-in-the-loop (07).
"""
import json
from typing import Annotated, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from guardrails import Guard, GuardrailBlocked, GuardrailConfig

from . import data, policies
from .tools import INJECTED, TOOL_MAP, TOOLSETS

AGENTS = ("billing", "technical", "general")
MAX_STEPS = 8      # specialist LLM calls per customer message
MAX_HANDOFFS = 2   # transfers per customer message
GUARD_CONFIG = GuardrailConfig()

BLOCKED_REPLY = ("I can't help with that request. If you have a question about an order, your account "
                 "or one of our products, I'm happy to help.")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupportState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    intent: str                     # triage decision for the conversation
    active_agent: str               # specialist currently handling the customer
    steps: int                      # specialist LLM calls this turn
    handoffs: int                   # transfers this turn
    pending_refund: Optional[dict]  # refund waiting for a supervisor
    status: str                     # open | blocked | awaiting_approval | refunded | escalated
    guard_notes: List[str]          # guardrail redactions/blocks this turn (for the UI)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRIAGE_PROMPT = (
    "You route customer messages for Acme Gadgets support. Choose one intent:\n"
    "- billing: orders, refunds, returns, charges, payments, plan/subscription changes\n"
    "- technical: devices not working, pairing, login errors, password resets, warranty repairs\n"
    "- general: shipping questions, store policies, opening hours, anything else\n"
    "- escalate: the customer explicitly asks for a human/manager, threatens legal action, or is extremely upset"
)

AGENT_PROMPTS = {
    "billing": "You are the billing specialist: orders, refunds, charges and plans.",
    "technical": "You are the technical specialist: device troubleshooting, login problems and warranty repairs.",
    "general": "You are the general support agent: shipping, policies, opening hours and general questions.",
}

SHARED_RULES = """
Customer: {name} ({plan} plan, member since {since}).
Previous cases for this customer:
{notes}

Rules:
- Answer from tool results and help-center articles only; cite the article title for policies. Never invent
  order details, prices, policies or timelines.
- You can only see this customer's orders. If an order isn't found, say so; never guess.
- For refunds, call request_refund and relay its message. Do not promise a refund before the tool confirms it.
- If the request belongs to another specialist, call transfer_to_agent. If the customer wants a person,
  is very upset, or you can't solve it, call escalate_to_human.
- Be warm and concise (2-4 sentences)."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class TriageDecision(BaseModel):
    intent: Literal["billing", "technical", "general", "escalate"]
    reason: str = Field(description="One short sentence")


_triage_llm = _llm.with_structured_output(TriageDecision)
_agent_llms = {name: _llm.bind_tools(TOOLSETS[name]) for name in AGENTS}


def _case_notes(store: BaseStore, customer_id: str) -> List[str]:
    items = store.search(("case_notes", customer_id), limit=20)
    return [i.value["note"] for i in sorted(items, key=lambda i: i.created_at)][-5:]


def _save_note(store: BaseStore, customer_id: str, note: str) -> None:
    existing = store.search(("case_notes", customer_id), limit=100)
    store.put(("case_notes", customer_id), f"note-{len(existing) + 1:04d}", {"note": note})


def _turn_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Messages since the latest customer message."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return messages[last_human:]

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def input_guard(state: SupportState) -> Command:
    last = state["messages"][-1]
    guard = Guard(GUARD_CONFIG)
    reset = {"steps": 0, "handoffs": 0, "status": "open", "guard_notes": []}
    try:
        text = guard.check_input(last.content)
    except GuardrailBlocked as exc:
        notes = [f"blocked: {v.check} ({v.detail})" for v in exc.violations]
        return Command(goto=END, update={**reset, "status": "blocked", "guard_notes": notes,
                                         "messages": [AIMessage(content=BLOCKED_REPLY)]})
    update = {**reset, "guard_notes": [f"{v.action}: {v.check} ({v.detail})" for _, v in guard.log]}
    if text != last.content:  # same id → add_messages replaces the original with the redacted text
        update["messages"] = [HumanMessage(content=text, id=last.id)]
    return Command(goto="triage", update=update)


def _transcript(messages: List[BaseMessage], turns: int = 6) -> str:
    """Plain-text recent conversation for triage (tool calls/results left out)."""
    lines = [f"{'Customer' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
             for m in messages if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content)]
    return "\n".join(lines[-turns:])


def triage(state: SupportState) -> Command:
    """Runs on every message, so the conversation moves to the right specialist when the topic changes."""
    decision = _triage_llm.invoke([
        SystemMessage(content=TRIAGE_PROMPT + "\nClassify the customer's LATEST message; earlier lines are context."),
        HumanMessage(content=_transcript(state["messages"])),
    ])
    if decision.intent == "escalate":
        return Command(goto="escalate", update={"intent": "escalate"})
    return Command(goto=decision.intent, update={"intent": decision.intent, "active_agent": decision.intent})


def escalate(state: SupportState, *, store: BaseStore) -> Command:
    reason = state["messages"][-1].content[:300]
    ticket = data.create_ticket(state["customer_id"], reason, priority="high")
    _save_note(store, state["customer_id"], f"Escalated to a human ({ticket['ticket_id']}): {reason[:80]}")
    reply = (f"I've passed this to a human agent — ticket {ticket['ticket_id']}. "
             "Someone will get back to you within 4 hours.")
    return Command(goto="output_guard", update={"status": "escalated", "messages": [AIMessage(content=reply)]})


def make_specialist(name: str):
    def specialist(state: SupportState, *, store: BaseStore) -> Command:
        if state.get("steps", 0) >= MAX_STEPS:
            return Command(goto="fallback")
        customer = data.get_customer(state["customer_id"])
        notes = _case_notes(store, state["customer_id"])
        system = AGENT_PROMPTS[name] + SHARED_RULES.format(
            name=customer["name"], plan=customer["plan"], since=customer["member_since"],
            notes="\n".join(f"- {n}" for n in notes) or "- none",
        )
        response = _agent_llms[name].invoke([SystemMessage(content=system), *state["messages"]])
        return Command(
            goto="tools" if response.tool_calls else "output_guard",
            update={"messages": [response], "steps": state.get("steps", 0) + 1},
        )
    specialist.__name__ = name
    return specialist


def tools(state: SupportState, *, store: BaseStore) -> Command:
    """Run the specialist's tool calls and decide where the graph goes next."""
    last, cid = state["messages"][-1], state["customer_id"]
    goto, update, results = state["active_agent"], {}, []
    earlier = {(c["name"], json.dumps(c["args"], sort_keys=True))
               for m in _turn_messages(state["messages"])[:-1] if isinstance(m, AIMessage) for c in m.tool_calls or []}

    for call in last.tool_calls:
        name, args = call["name"], dict(call["args"])
        if (name, json.dumps(call["args"], sort_keys=True)) in earlier:
            # Loop detection: the model is repeating itself instead of acting on the result
            content = ("You already made this exact call this turn; the result is above. Answer the customer, "
                       "call transfer_to_agent if another specialist should handle it, or escalate_to_human.")
        elif name == "transfer_to_agent":
            target = args.get("agent")
            if target in AGENTS and target != goto and state.get("handoffs", 0) < MAX_HANDOFFS:
                goto = target
                update.update(active_agent=target, handoffs=state.get("handoffs", 0) + 1)
                content = f"Transferred to the {target} specialist."
            else:
                content = "Transfer not possible; continue helping the customer yourself or escalate."
        elif name not in TOOL_MAP:
            content = f"Unknown tool '{name}'."
        else:
            if name in INJECTED:
                args["customer_id"] = cid  # from the session, never from the model
            content = TOOL_MAP[name].invoke(args)
            outcome = json.loads(content) if name in ("request_refund", "escalate_to_human") else {}
            if outcome.get("status") == "issued":
                update["status"] = "refunded"
                _save_note(store, cid, f"Refunded {outcome['order_id']} (${outcome['amount']:.2f}) automatically.")
            elif outcome.get("status") == "needs_approval":
                update.update(status="awaiting_approval", pending_refund=outcome)
                goto = "approval"
            elif outcome.get("status") == "escalated":
                update["status"] = "escalated"
                _save_note(store, cid, f"Escalated to a human ({outcome['ticket_id']}): {args.get('reason', '')[:80]}")
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))

    if goto == "approval":  # tell the customer before the graph pauses
        p = update["pending_refund"]
        results.append(AIMessage(content=(
            f"Your refund of ${p['amount']:.2f} for {p['item']} is eligible, and because it's over "
            f"${policies.AUTO_APPROVE_LIMIT:.0f} a supervisor needs to approve it. That usually takes under "
            "1 business day, and I'll update you right here.")))
    return Command(goto=goto, update={**update, "messages": results})


def approval(state: SupportState, *, store: BaseStore) -> Command:
    """Pause for a supervisor. Side effects happen only after interrupt() returns, because
    LangGraph re-runs this node from the top when the graph is resumed."""
    request = state["pending_refund"]
    decision = interrupt({"type": "refund_approval", "customer_id": state["customer_id"], **request})
    approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
    note = decision.get("note", "") if isinstance(decision, dict) else ""
    cid = state["customer_id"]

    # Re-check policy at execution time: the order may have changed while we waited
    verdict, why = policies.refund_verdict(data.get_order(request["order_id"], cid))
    if approved and verdict != "deny":
        refund = data.issue_refund(request["order_id"], cid, approved_by="supervisor")
        _save_note(store, cid, f"Refunded {request['order_id']} (${refund['amount']:.2f}) after supervisor approval.")
        reply = (f"Good news: a supervisor approved your refund of ${refund['amount']:.2f} for "
                 f"{request['item']} ({refund['refund_id']}). It will reach your original payment method "
                 "in 5-7 business days.")
        status = "refunded"
    else:
        ticket = data.create_ticket(cid, f"Refund for {request['order_id']} declined. {note}".strip(), priority="normal")
        _save_note(store, cid, f"Refund for {request['order_id']} declined by supervisor ({ticket['ticket_id']}).")
        detail = f" Reason: {note}" if note else (f" {why}" if verdict == "deny" else "")
        reply = (f"A supervisor reviewed your refund request for {request['item']} and couldn't approve it.{detail} "
                 f"I've opened ticket {ticket['ticket_id']} so a team member can follow up with options.")
        status = "escalated"
    return Command(goto="output_guard",
                   update={"pending_refund": None, "status": status, "messages": [AIMessage(content=reply)]})


def fallback(state: SupportState, *, store: BaseStore) -> Command:
    """Step budget exhausted — hand off to a human instead of looping."""
    ticket = data.create_ticket(state["customer_id"], "Assistant could not resolve the request", priority="normal")
    _save_note(store, state["customer_id"], f"Assistant hit its step limit; escalated ({ticket['ticket_id']}).")
    reply = (f"I'm sorry, I wasn't able to sort this out myself. I've created ticket {ticket['ticket_id']} "
             "and a human agent will follow up within 4 hours.")
    return Command(goto="output_guard", update={"status": "escalated", "messages": [AIMessage(content=reply)]})


def output_guard(state: SupportState) -> dict:
    last = state["messages"][-1]
    guard = Guard(GUARD_CONFIG)
    notes = list(state.get("guard_notes", []))
    try:
        text = guard.check_output(last.content)
    except GuardrailBlocked as exc:
        notes += [f"output blocked: {v.check}" for v in exc.violations]
        return {"guard_notes": notes, "messages": [AIMessage(content=BLOCKED_REPLY, id=last.id)]}
    notes += [f"output {v.action}: {v.check} ({v.detail})" for _, v in guard.log]
    update = {"guard_notes": notes}
    if text != last.content:
        update["messages"] = [AIMessage(content=text, id=last.id)]
    return update

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_app(checkpointer=None, store=None):
    """Compile a fresh graph (tests and evals get isolated memory)."""
    builder = StateGraph(SupportState)
    builder.add_node("input_guard", input_guard)
    builder.add_node("triage", triage)
    builder.add_node("escalate", escalate)
    for name in AGENTS:
        builder.add_node(name, make_specialist(name))
    builder.add_node("tools", tools)
    builder.add_node("approval", approval)
    builder.add_node("fallback", fallback)
    builder.add_node("output_guard", output_guard)
    builder.add_edge(START, "input_guard")
    builder.add_edge("output_guard", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver(), store=store or InMemoryStore())


app = build_app()

# ---------------------------------------------------------------------------
# Session API (used by the CLI demo, UI, evals and tests)
# ---------------------------------------------------------------------------

class SupportSession:
    """One customer conversation. send() returns the turn's result; when a refund needs
    approval the graph is paused and pending_approval holds the request until decide()."""

    def __init__(self, customer_id: str, thread_id: Optional[str] = None, graph=None):
        import uuid

        assert customer_id in data.CUSTOMERS, f"unknown customer {customer_id}"
        self.customer_id = customer_id
        self.graph = graph or app
        self.config = {"configurable": {"thread_id": thread_id or f"support-{uuid.uuid4().hex[:8]}"}}

    def send(self, text: str) -> dict:
        if self.pending_approval:
            # New input would start a fresh run and drop the paused approval, so hold the conversation
            p = self.pending_approval
            return {**self._turn(len(self.state["messages"])), "reply": (
                f"Your refund for {p['item']} is still with a supervisor. I'll reply here as soon as "
                "it's decided, and then I can help with anything else.")}
        before = len(self.state.get("messages", []))
        self.graph.invoke({"messages": [HumanMessage(content=text)], "customer_id": self.customer_id}, self.config)
        return self._turn(before)

    def decide(self, approved: bool, note: str = "") -> dict:
        assert self.pending_approval, "no refund is waiting for approval"
        before = len(self.state["messages"])
        self.graph.invoke(Command(resume={"approved": approved, "note": note}), self.config)
        return self._turn(before)

    @property
    def state(self) -> dict:
        return self.graph.get_state(self.config).values

    @property
    def pending_approval(self) -> Optional[dict]:
        interrupts = self.graph.get_state(self.config).interrupts
        return interrupts[0].value if interrupts else None

    def _turn(self, before: int) -> dict:
        state = self.state
        new = state["messages"][before:]
        ai = [m for m in new if isinstance(m, AIMessage)]
        return {
            "reply": next((m.content for m in reversed(ai) if m.content), ""),
            "status": state.get("status", "open"),
            "intent": state.get("intent"),
            "agent": state.get("active_agent"),
            "tools": [c["name"] for m in ai for c in (m.tool_calls or [])],
            "tool_results": [(m.name, m.content) for m in new if isinstance(m, ToolMessage)],
            "guard_notes": state.get("guard_notes", []),
            "pending_approval": self.pending_approval,
        }
