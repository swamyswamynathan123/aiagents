"""
Tools the support agents can call.

customer_id is an InjectedToolArg: it is hidden from the LLM's tool schema and
filled in by the graph from the authenticated session. The model can ask about
"order B-2001", but it cannot choose whose account to look in.

request_refund and escalate_to_human return JSON so the graph can act on the
outcome (pause for approval, record status); the agent just relays the message.
"""
import json
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from . import data, kb, policies

CustomerId = Annotated[str, InjectedToolArg]


def _order_line(o: dict) -> str:
    when = f"delivered {o['delivered']}" if o["delivered"] else f"ordered {o['ordered']}"
    extra = f", tracking {o['tracking']}" if o["tracking"] else ""
    refunded = ", refunded" if o["refunded"] else ""
    return f"{o['order_id']}: {o['item']} (${o['amount']:.2f}) — {o['status']}, {when}{extra}{refunded}"


@tool
def list_my_orders(customer_id: CustomerId) -> str:
    """List the current customer's orders with status, dates and tracking."""
    orders = data.list_orders(customer_id)
    return "\n".join(_order_line(o) for o in orders) or "No orders on this account."


@tool
def get_order(order_id: str, customer_id: CustomerId) -> str:
    """Look up one of the current customer's orders by its ID (e.g. A-1001)."""
    order = data.get_order(order_id, customer_id)
    return _order_line(order) if order else f"No order {order_id} found on this account."


@tool
def request_refund(order_id: str, reason: str, customer_id: CustomerId) -> str:
    """Request a refund for one of the customer's orders. Policy decides whether it is issued now,
    sent to a supervisor for approval, or declined — relay the returned message to the customer."""
    order = data.get_order(order_id, customer_id)
    verdict, why = policies.refund_verdict(order)
    if verdict == "deny":
        return json.dumps({"status": "denied", "order_id": order_id, "message": why})
    if verdict == "auto_approve":
        refund = data.issue_refund(order["order_id"], customer_id, approved_by="policy")
        return json.dumps({"status": "issued", **refund,
                           "message": f"Refund {refund['refund_id']} of ${refund['amount']:.2f} issued; "
                                      "it arrives in 5-7 business days."})
    return json.dumps({"status": "needs_approval", "order_id": order["order_id"], "item": order["item"],
                       "amount": order["amount"], "reason": reason, "message": why})


@tool
def search_help_center(query: str) -> str:
    """Search the help-center articles. Cite the article title in your answer."""
    return "\n\n".join(f"[{title}] {text}" for title, text in kb.search(query))


@tool
def escalate_to_human(reason: str, customer_id: CustomerId) -> str:
    """Create a ticket for a human agent. Use when the customer asks for a person, is very upset,
    or the issue can't be solved with the available tools."""
    ticket = data.create_ticket(customer_id, reason, priority="high")
    return json.dumps({"status": "escalated", **ticket,
                       "message": f"Ticket {ticket['ticket_id']} created; a human agent will reply within 4 hours."})


@tool
def transfer_to_agent(agent: Literal["billing", "technical", "general"], reason: str) -> str:
    """Hand the conversation to another specialist when the request is outside your area."""
    return f"Transferring to {agent}."  # the graph performs the actual handoff


TOOLSETS = {
    "billing": [list_my_orders, get_order, request_refund, search_help_center, transfer_to_agent, escalate_to_human],
    "technical": [search_help_center, get_order, transfer_to_agent, escalate_to_human],
    "general": [search_help_center, list_my_orders, get_order, transfer_to_agent, escalate_to_human],
}
TOOL_MAP = {t.name: t for tools in TOOLSETS.values() for t in tools}
INJECTED = {"list_my_orders", "get_order", "request_refund", "escalate_to_human"}  # take customer_id
