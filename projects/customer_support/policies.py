"""
Refund policy as plain code.

Business rules live here, not in prompts: the LLM decides *that* a customer is
asking for a refund, but whether one is allowed (and who must approve it) is
deterministic, testable and the same every time.
"""
from datetime import date
from typing import Literal, Optional, Tuple

REFUND_WINDOW_DAYS = 30
AUTO_APPROVE_LIMIT = 50.00  # refunds above this need a human supervisor

Verdict = Literal["auto_approve", "needs_approval", "deny"]


def refund_verdict(order: Optional[dict], today: Optional[date] = None) -> Tuple[Verdict, str]:
    """Decide what happens to a refund request. Returns (verdict, customer-facing reason)."""
    today = today or date.today()
    if order is None:
        return "deny", "That order isn't on this account."
    if order["refunded"]:
        return "deny", "This order has already been refunded."
    if order["status"] != "delivered":
        return "deny", (f"The order is still {order['status']}. Refunds are available once it's delivered; "
                        "you can also refuse the delivery and it will be refunded automatically.")
    days = (today - date.fromisoformat(order["delivered"])).days
    if days > REFUND_WINDOW_DAYS:
        return "deny", (f"It was delivered {days} days ago, outside the {REFUND_WINDOW_DAYS}-day refund window. "
                        "It may still be covered by the 1-year warranty.")
    if order["amount"] <= AUTO_APPROVE_LIMIT:
        return "auto_approve", "Eligible and within the automatic refund limit."
    return "needs_approval", (f"Eligible, but refunds over ${AUTO_APPROVE_LIMIT:.0f} are reviewed by a supervisor.")
