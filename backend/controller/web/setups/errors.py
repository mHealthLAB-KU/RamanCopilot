"""Converts tool results and exceptions into HTTP responses."""

from __future__ import annotations

import traceback

from fastapi import HTTPException


def unwrap(result, *, on_error: int = 500):
    """Return a tool result, converting ok=False into an HTTP error (409 if the instrument is busy)."""
    if isinstance(result, dict) and result.get("ok") is False:
        if result.get("busy_with"):
            raise HTTPException(status_code=409, detail=result.get("error", "instrument busy"))
        raise HTTPException(status_code=on_error,
                            detail=result.get("error", "the request could not be processed"))
    return result


def http_error(e: Exception, *, status_code: int = 500, detail: str | None = None) -> HTTPException:
    """Convert an exception into an HTTPException, logging the stack trace on the server."""
    traceback.print_exc()
    return HTTPException(status_code=status_code,
                         detail=detail or f"{type(e).__name__}: {e}")
