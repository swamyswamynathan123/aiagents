"""
Number grounding: every figure in the final answer must appear in some execution output.

This is the deterministic half of the reflection step. The LLM critic judges the method;
this code catches the most damaging failure, a number the model wrote but never computed.
Matching tolerates the formatting models use: thousands separators, currency, %, rounding to
the shown precision, and K/M/B or "thousand/million/billion" suffixes.
"""
import re
from typing import Iterable, List, Tuple

_NUM = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(%|[kKmMbB]\b|thousand|million|billion)?"
)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
_DATES = re.compile(  # 2025-12-31, 12/31/2025, December 31, Dec 31st: dates aren't claims about the data
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


def extract(text: str) -> List[Tuple[str, float, float]]:
    """Return (as_written, value, tolerance) for each number in the text (dates are skipped)."""
    found = []
    text = _DATES.sub(" ", text)
    for m in _NUM.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        digits = raw.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = _SCALE.get(unit, 1.0)
        value = float(digits) * scale
        found.append((m.group(0).strip(), value, 0.5 * 10 ** -decimals * scale))
    return found


def _ignorable(written: str, value: float) -> bool:
    """Small counts, years and the like aren't claims about the data."""
    if "%" in written or any(c.isalpha() for c in written):
        return False
    return (value.is_integer() and 0 <= value <= 12) or (value.is_integer() and 1900 <= value <= 2100 and "," not in written)


def ungrounded_numbers(answer: str, outputs: Iterable[str], question: str = "") -> List[str]:
    """Numbers in the answer that don't match any printed output value (within display rounding)."""
    candidates = [v for text in outputs for _, v, _ in extract(text)]
    mentioned = {v for _, v, _ in extract(question)}
    missing = []
    for written, value, tol in extract(answer):
        if _ignorable(written, value) or value in mentioned:
            continue
        targets = [value, value / 100] if written.endswith("%") else [value]  # 12.5% may be printed as 0.125
        tol_for = lambda t: tol if t == value else tol / 100  # noqa: E731
        if not any(abs(c - t) <= max(tol_for(t), 1e-9) or abs(c - t) <= abs(t) * 1e-6
                   for c in candidates for t in targets):
            missing.append(written)
    return missing
