"""Machine-readable JSON log of each agent turn."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from backend.repo.store import LOG_ROOT
from backend.repo.store.run_store import existing_log, log_stem

LOG_DIR = LOG_ROOT

_SESSION_FILES: dict[tuple[str, str], Path] = {}

_LOCK = threading.Lock()

_MAX_RESULT_CHARS = 4000  # characters kept per tool result

_MAX_ERROR_CHARS = 2000  # characters kept per error message

_MAX_TRACEBACK_CHARS = 4000  # characters kept from the end of a traceback


def _cap(obj: Any) -> Any:
    """Truncate an over-long serialization, marking that it was truncated."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(obj)[:_MAX_RESULT_CHARS]
    if len(s) <= _MAX_RESULT_CHARS:
        return obj
    return {"_truncated": True, "chars": len(s), "preview": s[:_MAX_RESULT_CHARS]}


def status_of(exc: BaseException) -> Optional[int]:
    """Extract the HTTP status code from an exception, or None."""
    for attr in ("status_code", "status", "code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    inner = getattr(exc, "response", None)
    v = getattr(inner, "status_code", None)
    if isinstance(v, int):
        return v
    m = re.search(r"\b(4\d{2}|5\d{2})\b", str(exc))
    return int(m.group(1)) if m else None


def error_info(exc: BaseException) -> dict:
    """Return the full error information of an exception, including the traceback."""
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:_MAX_ERROR_CHARS],
        "status": status_of(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-_MAX_TRACEBACK_CHARS:],
    }


def _flat_llm(res: dict, call_took_s: float) -> dict:
    """Flatten an LLMResult into one log record."""
    out = {
        "actual_model": res.get("actual_model", ""),
        "request_id": res.get("request_id", ""),
        "call_took_s": round(float(call_took_s), 4),
        "input_tokens": int(res.get("input_tokens") or 0),
        "output_tokens": int(res.get("output_tokens") or 0),
        "cached_tokens": int(res.get("cached_tokens") or 0),
        "response_chars": len(res.get("text") or ""),
        "thinking_chars": len(res.get("thinking_text") or ""),
        "thinking_tokens": int(res.get("thinking_tokens") or 0),
        "stop_reason": res.get("stop_reason", ""),
        "is_truncated": bool(res.get("is_truncated")),
        "is_blocked": bool(res.get("is_blocked")),
    }
    out.update(res.get("timing") or {})
    return out


class RunLog:
    """Log of one turn: create, feed, close."""

    def __init__(self, agent: str, session_id: str, question: str):
        self.agent = str(agent or "")
        self.session_id = str(session_id or "")
        self.question = str(question or "")
        self._t0 = time.perf_counter()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.tool_calls: list[dict] = []
        self.llm_calls: list[dict] = []
        self._closed = False

    def llm(self, res: dict, call_took_s: float, **retry: Any) -> None:
        """Record one LLM response."""
        entry = {"step": len(self.llm_calls)}
        entry.update(_flat_llm(res, call_took_s))
        entry.update(retry)
        self.llm_calls.append(entry)

    def llm_failed(self, exc: BaseException, call_took_s: float = 0.0, **retry: Any) -> None:
        """Record an LLM call that failed after all retries."""
        self.llm_calls.append({
            "step": len(self.llm_calls),
            "call_took_s": round(float(call_took_s), 4),
            "error": error_info(exc),
            **retry,
        })

    def tool_failed(self, name: str, args: dict, exc: BaseException,
                    took_ms: float = 0.0) -> None:
        """Record a tool that raised an exception."""
        self.tool_calls.append({
            "step": len(self.tool_calls),
            "name": name,
            "ms": round(float(took_ms), 1),
            "ok": False,
            "error": error_info(exc),
            "args": _cap(args),
            "img_chars": None,
            "result": None,
        })

    def tool(self, name: str, args: dict, ex: dict) -> None:
        """Record one tool execution."""
        result = ex.get("result")
        img = ex.get("img_b64") or ""
        self.tool_calls.append({
            "step": len(self.tool_calls),
            "name": name,
            "tool_call_id": ex.get("tool_call_id") or None,
            "ms": round(float(ex.get("elapsed_ms") or 0.0), 1),
            "ok": result.get("ok", True) if isinstance(result, dict) else True,
            "error": result.get("error") if isinstance(result, dict) else None,
            "args": _cap(args),
            "img_chars": len(img) or None,
            "result": _cap(result),
        })

    def close(self, answer: str = "", ctx: Optional[dict] = None,
              error: Any = None, response_type: str = "done") -> None:
        """Close the turn and append it to the log file."""
        if self._closed:
            return
        self._closed = True
        ctx = ctx or {}
        if isinstance(error, BaseException):
            error = error_info(error)
        entry = {
            "turn_index": None,
            "started_at": self.started_at,
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "turn_took_s": round(time.perf_counter() - self._t0, 3),
            "question": self.question,
            "answer": answer,
            "response_type": response_type,
            "error": error,
            "n_llm_calls": len(self.llm_calls),
            "n_tool_calls": len(self.tool_calls),
            "n_tool_failed": sum(1 for c in self.tool_calls if not c.get("ok", True)),
            "tool_call_order": list(ctx.get("tool_call_order") or []),
            "laser_shots": list(ctx.get("laser_shots") or []),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "reflection": ctx.get("reflection"),
        }
        try:
            _append(self.agent, self.session_id, self.started_at, entry)
        except Exception as e:  # noqa: BLE001
            print(f"[run_log] write failed: {type(e).__name__}: {e}", file=sys.stderr)

    def fail(self, detail: Any, ctx: Optional[dict] = None) -> None:
        """Close the turn as an error, keeping everything collected so far."""
        self.close(answer="", ctx=ctx, error=detail, response_type="error")


def _session_file(agent: str, session_id: str, started_at: str) -> Path:
    """Return the log file for this (agent, session), chosen on first call and cached."""
    key = (agent, session_id)
    hit = _SESSION_FILES.get(key)
    if hit is not None:
        return hit
    digits = "".join(c for c in started_at if c.isdigit())
    stamp = f"{digits[:8]}_{digits[8:14]}" if len(digits) >= 14 else digits
    path = (existing_log(LOG_DIR, agent, session_id, ".json")
            or LOG_DIR / f"{log_stem(agent, stamp, session_id)}.json")
    _SESSION_FILES[key] = path
    return path


def _append(agent: str, session_id: str, started_at: str, entry: dict) -> None:
    """Append a turn to the session file by rewriting it as valid JSON."""
    with _LOCK:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _session_file(agent, session_id, started_at)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or "turns" not in doc:
                raise ValueError("unexpected shape")
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            doc = {"agent": agent, "session_id": session_id,
                   "started_at": started_at, "turns": []}
        entry["turn_index"] = len(doc["turns"])
        doc["turns"].append(entry)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
