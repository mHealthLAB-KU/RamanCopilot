"""Per-session tool state."""

from __future__ import annotations

import threading


_SESSION_STATE: dict[str, dict] = {}
_SESSION_STATE_LOCK = threading.Lock()
_SESSION_STATE_MAX = 16  # sessions kept in memory


def _new_session_state() -> dict:
    return {
        "last_spectrum": None,
        "grid_gate":     {"geom": None, "resolved": None, "state": "none", "enforce": False},
    }


def _sstate() -> dict:
    """Return the tool state of the current session."""
    try:
        from backend.repo.store import run_store
        key = run_store.current().get("label") or "_unassigned"
    except Exception:
        key = "_unassigned"
    with _SESSION_STATE_LOCK:
        st = _SESSION_STATE.get(key)
        if st is None:
            if len(_SESSION_STATE) >= _SESSION_STATE_MAX:
                _SESSION_STATE.pop(next(iter(_SESSION_STATE)), None)
            st = _new_session_state()
            _SESSION_STATE[key] = st
        return st


def _cache_and_return(result: dict) -> dict:
    """Cache an acquire_spectrum() result for this session and return it unchanged."""
    if result.get("ok") and "data" in result:
        _sstate()["last_spectrum"] = result
    return result
