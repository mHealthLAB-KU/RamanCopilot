"""Common response format for all tools."""
from __future__ import annotations

from typing import Any

_REPR_LIMIT = 200


def ok(**payload: Any) -> dict:
    """Build a success result {"ok": True, **payload}."""
    return {"ok": True, **payload}


def fail(error: Any, **extra: Any) -> dict:
    """Build a failure result {"ok": False, "error": ..., **extra}."""
    return {"ok": False, "error": "" if error is None else str(error), **extra}


def is_ok(result: Any) -> bool:
    """Return True only if the result is a dict with ok exactly True."""
    return isinstance(result, dict) and result.get("ok") is True


def normalize(result: Any, tool: str) -> dict:
    """Coerce a tool return value into the response format; violations become model-visible failures."""
    if not isinstance(result, dict):
        return fail(f"{tool} returned {type(result).__name__}, expected a dict: "
                    f"{repr(result)[:_REPR_LIMIT]}")

    if not isinstance(result.get("ok"), bool):
        return fail(f"{tool} returned a malformed result (no boolean 'ok' field): "
                    f"{repr(result)[:_REPR_LIMIT]}")

    err = result.get("error")
    if not result["ok"] and (err is None or not str(err).strip()):
        return {**result,
                "error": f"{tool} reported failure without an error message. "
                         f"Look at the other fields of this result for the reason."}
    return result


__all__ = ["ok", "fail", "is_ok", "normalize"]
