"""Parses provider responses into one common result format."""

from __future__ import annotations

import importlib
from typing import Any, TypedDict

from backend.repo.llm.providers import parse_model


class LLMResult(TypedDict, total=False):
    """Provider-independent result of one LLM call."""
    provider: str
    actual_model: str
    request_id: str

    text: str
    tool_calls: list[dict[str, Any]]
    history: Any
    server_tools: list[dict[str, Any]]

    thinking_text: str
    thinking_tokens: int

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int

    stop_reason: str
    is_truncated: bool
    is_blocked: bool

    timing: dict[str, float]


EMPTY: LLMResult = {
    "provider": "", "actual_model": "", "request_id": "",
    "text": "", "tool_calls": [], "history": "", "server_tools": [],
    "thinking_text": "", "thinking_tokens": 0,
    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0,
    "stop_reason": "", "is_truncated": False, "is_blocked": False,
    "timing": {},
}


def _blocks(ai) -> list[dict]:
    """Return the dict blocks of a block-list content, or an empty list for string content."""
    c = getattr(ai, "content", None)
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def _text_of(ai) -> str:
    """Return the answer text only (thinking blocks excluded)."""
    c = getattr(ai, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        return "".join(out)
    return str(c or "")


def _thinking_blocks(ai) -> str:
    """Concatenate the thinking blocks (Anthropic and Google)."""
    return "".join(str(b.get("thinking") or "")
                   for b in _blocks(ai) if b.get("type") == "thinking")


def _snake(name: str) -> str:
    """modelVersion -> model_version."""
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name)


def _pick(meta: dict, keys) -> str:
    """Return the first metadata value among the field names declared by the DTO."""
    for k in keys:
        for cand in (k, _snake(k)):
            if meta.get(cand):
                return str(meta[cand])
    return ""


def _sec(ns) -> float:
    """Convert nanoseconds to seconds."""
    try:
        return round(float(ns or 0) / 1e9, 4)
    except Exception:
        return 0.0


def _module(provider: str, model: str):
    """Return the DTO module describing this response."""
    mod = importlib.import_module(f"backend.repo.llm.llm_response.{provider}")
    sub = getattr(mod, "MODULE", None)
    if sub and model in sub:
        return importlib.import_module(
            f"backend.repo.llm.llm_response.{provider}.{sub[model]}")
    return mod


def _common(ai, meta: dict, usage: dict, dto) -> LLMResult:
    return LLMResult(
        text=_text_of(ai),
        tool_calls=list(getattr(ai, "tool_calls", None) or []),
        thinking_tokens=int((usage.get("output_token_details") or {}).get("reasoning") or 0),
        input_tokens=int(usage.get("input_tokens") or meta.get("prompt_eval_count") or 0),
        output_tokens=int(usage.get("output_tokens") or meta.get("eval_count") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        cached_tokens=int((usage.get("input_token_details") or {}).get("cache_read") or 0),
        actual_model=_pick(meta, getattr(dto, "SERVED_KEYS", ())),
        request_id=_pick(meta, getattr(dto, "REQUEST_ID_KEYS", ())),
    )


def _from_ollama(ai, meta: dict, usage: dict) -> LLMResult:
    """Extract fields from an Ollama response."""
    return LLMResult(
        thinking_text=(getattr(ai, "additional_kwargs", None) or {}).get("reasoning_content") or "",
        stop_reason=str(meta.get("done_reason") or ""),
        timing={k: _sec(v) for k, v in (
            ("server_load_s", meta.get("load_duration")),
            ("server_prefill_s", meta.get("prompt_eval_duration")),
            ("server_decode_s", meta.get("eval_duration")),
            ("server_total_s", meta.get("total_duration")),
        ) if v is not None},
    )


def _from_openai(ai, meta: dict, usage: dict) -> LLMResult:
    """Extract fields from an OpenAI response (Chat Completions or Responses API)."""
    inc = meta.get("incomplete_details")
    if isinstance(inc, dict) and inc.get("reason"):
        stop_reason = str(inc["reason"])
    else:
        stop_reason = str(meta.get("finish_reason") or meta.get("status") or "")

    parts = []
    for b in _blocks(ai):
        if b.get("type") != "reasoning":
            continue
        for s in b.get("summary") or []:
            parts.append(str(s.get("text") or "") if isinstance(s, dict) else str(s))

    searches = []
    for b in _blocks(ai):
        act = b.get("action") or {}
        if b.get("type") != "web_search_call" or not (act.get("queries") or act.get("query")):
            continue
        searches.append(_search_record(
            act.get("queries") or ([act["query"]] if act.get("query") else []),
            [s for s in act.get("sources") or [] if isinstance(s, dict)],
            ok=b.get("status") == "completed"))

    return LLMResult(thinking_text="".join(parts), stop_reason=stop_reason,
                     server_tools=searches)


def _from_anthropic(ai, meta: dict, usage: dict) -> LLMResult:
    """Extract fields from an Anthropic response."""
    blocks = _blocks(ai)
    queries = {b.get("id"): (b.get("input") or {}).get("query", "")
               for b in blocks if b.get("type") == "server_tool_use"}
    searches = []
    for b in blocks:
        if b.get("type") != "web_search_tool_result":
            continue
        c = b.get("content")
        q = [queries.get(b.get("tool_use_id"), "")]
        if isinstance(c, list):
            searches.append(_search_record(q, [r for r in c if isinstance(r, dict)]))
        else:
            searches.append(_search_record(q, [], ok=False,
                                           error=str((c or {}).get("error_code") or "error")))
    return LLMResult(
        thinking_text=_thinking_blocks(ai),
        stop_reason=str(meta.get("stop_reason") or ""),
        server_tools=searches,
    )


def _from_google(ai, meta: dict, usage: dict) -> LLMResult:
    """Extract fields from a Google response."""
    gm = meta.get("grounding_metadata") or {}
    queries = list(gm.get("web_search_queries") or [])
    searches = [_search_record(
        queries, [{"url": (ch.get("web") or {}).get("uri", ""),
                   "title": (ch.get("web") or {}).get("title", "")}
                  for ch in gm.get("grounding_chunks") or [] if isinstance(ch, dict)])] if queries else []
    return LLMResult(
        thinking_text=_thinking_blocks(ai),
        stop_reason=str(meta.get("finish_reason") or ""),
        server_tools=searches,
    )


def _search_record(queries: list, sources: list, ok: bool = True, error: str = "") -> dict:
    """Convert one server-side search into the same log record shape as the web_search tool."""
    results = [{k: s[k] for k in ("url", "title") if s.get(k)} for s in sources]
    result: dict = {"ok": ok, "server_side": True, "results": [r for r in results if r.get("url")]}
    if error:
        result["error"] = error
    return {"name": "web_search", "args": {"query": " | ".join(q for q in queries if q)},
            "result": result}


_KEEP_IN_HISTORY = {
    "anthropic": (frozenset({"server_tool_use", "web_search_tool_result"}),
                  frozenset({"thinking", "redacted_thinking"})),
    "openai": (frozenset({"web_search_call"}), frozenset()),
}


def _history(ai, text: str, provider: str) -> Any:
    """Return the content to store in history (block list only when server search blocks are present)."""
    keep, drop = _KEEP_IN_HISTORY.get(provider, (frozenset(), frozenset()))
    blocks = getattr(ai, "content", None)
    if not isinstance(blocks, list):
        return text
    answered = {b.get("tool_use_id") for b in blocks
                if isinstance(b, dict) and b.get("type") == "web_search_tool_result"}
    blocks = [b for b in blocks
              if not (isinstance(b, dict) and b.get("type") == "server_tool_use" and b.get("id") not in answered)]
    if not any(isinstance(b, dict) and b.get("type") in keep for b in blocks):
        return text
    return [b for b in blocks if not (isinstance(b, dict) and b.get("type") in drop)]


ADAPTERS = {
    "ollama": _from_ollama,
    "openai": _from_openai,
    "anthropic": _from_anthropic,
    "google": _from_google,
}


def parse(ai, spec: str = "") -> LLMResult:
    """Convert a LangChain AIMessage into the common response format."""
    if not spec:
        import backend.common.config.llm as cfg
        spec = cfg.OLLAMA_MODEL
    provider, model = parse_model(spec)

    meta = getattr(ai, "response_metadata", None) or {}
    usage = getattr(ai, "usage_metadata", None) or {}
    dto = _module(provider, model)

    out: LLMResult = {**EMPTY,
                      **_common(ai, meta, usage, dto),
                      **ADAPTERS[provider](ai, meta, usage)}

    out["provider"] = provider
    out["actual_model"] = out["actual_model"] or model
    out["history"] = _history(ai, out["text"], provider)
    out["is_truncated"] = out["stop_reason"] in getattr(dto, "TRUNCATED", frozenset())
    out["is_blocked"] = out["stop_reason"] in getattr(dto, "BLOCKED", frozenset())
    return out


def _all_dtos():
    """Return all registered response DTO modules."""
    from backend.repo.llm.providers import PROVIDERS
    for prov in PROVIDERS:
        mod = importlib.import_module(f"backend.repo.llm.llm_response.{prov}")
        yield mod
        for sub in sorted(set((getattr(mod, "MODULE", None) or {}).values())):
            yield importlib.import_module(f"backend.repo.llm.llm_response.{prov}.{sub}")


def truncated_reasons() -> frozenset[str]:
    """Return every stop-reason spelling that means the output limit was hit."""
    return frozenset().union(*(m.TRUNCATED for m in _all_dtos()))


__all__ = ["LLMResult", "EMPTY", "ADAPTERS", "parse", "truncated_reasons"]
