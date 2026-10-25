"""Web search tool for Ollama models."""
from __future__ import annotations

import os

import httpx

from backend.common.result import fail, ok

_URL = "https://ollama.com/api/web_search"
_TIMEOUT = 15.0  # seconds
_MAX_CONTENT_CHARS = 1500  # characters kept from each result page


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web and return the top results (title, URL, snippet)."""
    query = (query or "").strip()
    if not query:
        return fail("The search query is empty.")
    max_results = max(1, min(int(max_results or 5), 10))

    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        return fail("Web search is not configured (OLLAMA_API_KEY is not set). "
                    "Answer from your own knowledge and say the web could not be checked.")
    try:
        r = httpx.post(_URL, json={"query": query, "max_results": max_results},
                       headers={"Authorization": f"Bearer {key}"}, timeout=_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("results") or []
    except httpx.HTTPStatusError as e:
        return fail(f"Web search failed (HTTP {e.response.status_code}). "
                    "If this persists, answer from your own knowledge and say the web could not be checked.")
    except httpx.HTTPError as e:
        return fail(f"Web search failed (network): {type(e).__name__}. "
                    "Check that the instrument PC is connected to the internet. "
                    "If offline, answer from your own knowledge and say the web could not be checked.")
    except Exception as e:
        return fail(f"Web search failed: {e}")

    results = [{"title": it.get("title", ""), "url": it.get("url", ""),
                "snippet": (it.get("content") or "")[:_MAX_CONTENT_CHARS]}
               for it in items[:max_results]]
    out = ok(provider="ollama", query=query, results=results)
    if not results:
        out["note"] = "No web results. Rephrase the query."
    return out
