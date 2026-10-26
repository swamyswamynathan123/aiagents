"""
Help-center knowledge base + retrieval (RAG).

Articles are embedded into an in-memory FAISS index on first search. Results keep
the article title so agents can cite their source.
"""
import threading
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

ARTICLES = [
    ("Refund policy",
     "Delivered items can be refunded within 30 days of delivery. Refunds go back to the original payment method "
     "within 5-7 business days. Refunds over $50 are reviewed by a supervisor, usually within 1 business day. "
     "Items still in transit cannot be refunded yet; refuse the delivery and the order is refunded automatically."),
    ("Damaged or defective items",
     "If an item arrives damaged, you can request a refund within 30 days of delivery or a free replacement. "
     "Keep the packaging until the case is resolved."),
    ("Warranty",
     "All Acme Gadgets products include a 1-year limited warranty covering manufacturing defects. "
     "Warranty claims outside the refund window are handled by the technical team as a repair or replacement."),
    ("Shipping and tracking",
     "Standard shipping takes 3-5 business days; express takes 1-2. Tracking numbers appear on the order once it "
     "ships. Orders over $35 ship free."),
    ("Resetting your password",
     "Go to Sign in > Forgot password and enter your account email. The reset link expires after 30 minutes. "
     "If the email doesn't arrive, check spam or contact support."),
    ("Login error 403",
     "A 403 error at sign-in usually means an expired session. Clear cookies for acmegadgets.example, then sign in "
     "again. If you use two-factor authentication, make sure your device clock is set automatically."),
    ("Pairing wireless earbuds",
     "Place both earbuds in the case, hold the case button for 5 seconds until the light flashes white, then select "
     "'Acme Buds' in your phone's Bluetooth settings. To reset, hold the button for 15 seconds."),
    ("Plans and billing",
     "Acme Care Basic is free. Acme Care Pro costs $9.99 per month and adds express shipping and priority support. "
     "You can cancel Pro at any time from Account > Plan; it stays active until the end of the billing period."),
    ("Support hours and contact",
     "Chat support is available 24/7. Phone support runs Monday to Friday, 9am to 6pm Eastern Time. "
     "Pro members get a priority phone line."),
]

_index: Optional[FAISS] = None
_lock = threading.Lock()


def _get_index() -> FAISS:
    global _index
    with _lock:
        if _index is None:
            _index = FAISS.from_texts(
                [f"{title}: {body}" for title, body in ARTICLES],
                embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
                metadatas=[{"title": title} for title, _ in ARTICLES],
            )
    return _index


def search(query: str, k: int = 3) -> List[Tuple[str, str]]:
    """Return the k most relevant (title, text) pairs."""
    return [(d.metadata["title"], d.page_content) for d in _get_index().similarity_search(query, k=k)]
