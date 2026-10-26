"""
Web search for the researcher nodes, with untrusted-content handling.

Search results are third-party text that ends up inside an LLM prompt, so they
are treated as untrusted: each result is truncated, and any result containing a
prompt-injection pattern is dropped before the model ever sees it (indirect
prompt injection). Dropped sources are reported so the UI and evals can show them.
"""
from typing import List, Tuple

from langchain_tavily import TavilySearch

from guardrails import detect_injection_heuristic

MAX_RESULTS = 4
MAX_CHARS_PER_SOURCE = 1200

_tavily = TavilySearch(max_results=MAX_RESULTS, include_answer=False)


def search(query: str) -> Tuple[List[dict], List[dict]]:
    """Return (sources, blocked). Each source is {"title", "url", "content"}."""
    response = _tavily.invoke({"query": query})
    if not isinstance(response, dict) or "error" in response:
        return [], []  # no results or search failure: the researcher reports the gap
    sources, blocked = [], []
    for r in response.get("results", []):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        reason = detect_injection_heuristic(content)
        if reason:
            blocked.append({"title": r.get("title", ""), "url": r.get("url", ""), "reason": reason})
            continue
        sources.append({"title": r.get("title") or r.get("url", ""), "url": r.get("url", ""),
                        "content": content[:MAX_CHARS_PER_SOURCE]})
    return sources, blocked
