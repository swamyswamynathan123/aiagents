"""
Mock backend for the support assistant: customers, orders, refunds and tickets.

Everything is in memory and re-seeded by reset(). Order dates are relative to
today so the refund-window rules give the same answers whenever the demo runs.
Every lookup takes the customer_id: the backend, not the LLM, enforces that a
customer can only see their own orders.
"""
import threading
from copy import deepcopy
from datetime import date, timedelta
from typing import Dict, List, Optional

CUSTOMERS: Dict[str, dict] = {
    "C001": {"name": "Alice Chen", "plan": "Pro", "member_since": "2023"},
    "C002": {"name": "Bob Martin", "plan": "Basic", "member_since": "2025"},
}


def _seed_orders() -> Dict[str, dict]:
    today = date.today()

    def order(customer_id, item, amount, status, days_ago, tracking=None):
        return {
            "customer_id": customer_id, "item": item, "amount": amount, "status": status,
            "ordered": (today - timedelta(days=days_ago + 4)).isoformat(),
            "delivered": (today - timedelta(days=days_ago)).isoformat() if status == "delivered" else None,
            "tracking": tracking, "refunded": False,
        }

    return {
        "A-1001": order("C001", "Wireless Earbuds", 39.99, "delivered", 5),
        "A-1002": order("C001", "27-inch 4K Monitor", 329.00, "delivered", 10),
        "A-1003": order("C001", "USB-C Hub", 24.99, "shipped", 1, tracking="TRK-48213-US"),
        "A-1004": order("C001", "Mechanical Keyboard", 89.00, "delivered", 45),
        "B-2001": order("C002", "Smart Watch", 199.00, "delivered", 3),
    }


_lock = threading.Lock()
ORDERS: Dict[str, dict] = {}
REFUNDS: List[dict] = []
TICKETS: List[dict] = []


def reset() -> None:
    """Restore the seed data (tests and evals call this between scenarios)."""
    with _lock:
        ORDERS.clear()
        ORDERS.update(_seed_orders())
        REFUNDS.clear()
        TICKETS.clear()


reset()


def get_customer(customer_id: str) -> Optional[dict]:
    return CUSTOMERS.get(customer_id)


def get_order(order_id: str, customer_id: str) -> Optional[dict]:
    """Return the order only if it belongs to this customer."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None or order["customer_id"] != customer_id:
        return None
    return {"order_id": order_id.strip().upper(), **deepcopy(order)}


def list_orders(customer_id: str) -> List[dict]:
    return [{"order_id": oid, **deepcopy(o)} for oid, o in ORDERS.items() if o["customer_id"] == customer_id]


def issue_refund(order_id: str, customer_id: str, approved_by: str) -> dict:
    with _lock:
        order = ORDERS[order_id]
        assert order["customer_id"] == customer_id and not order["refunded"]
        order["refunded"] = True
        refund = {"refund_id": f"R-{len(REFUNDS) + 1:04d}", "order_id": order_id, "customer_id": customer_id,
                  "amount": order["amount"], "approved_by": approved_by}
        REFUNDS.append(refund)
    return refund


def create_ticket(customer_id: str, reason: str, priority: str = "normal") -> dict:
    with _lock:
        ticket = {"ticket_id": f"T-{len(TICKETS) + 1:04d}", "customer_id": customer_id,
                  "reason": reason, "priority": priority}
        TICKETS.append(ticket)
    return ticket
