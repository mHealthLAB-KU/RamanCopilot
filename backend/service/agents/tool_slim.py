"""Reduces tool results before they are added to the model context."""

from __future__ import annotations

from typing import Any

KEEP_KEYS = {"files", "artifacts", "saved_files", "rules"}

MAX_SCALAR_LIST = 32  # longer lists of numbers are replaced by a summary

MAX_RECORDS = 60  # records kept from a list of records

_MAX_DEPTH = 4  # recursion depth


def _is_record_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(e, dict) for e in v)


def slim(result: Any, _depth: int = 0) -> Any:
    """Recursively replace raw arrays in a tool result with summaries."""
    if _depth >= _MAX_DEPTH or not isinstance(result, dict):
        return result

    out: dict = {}
    for k, v in result.items():
        if k in KEEP_KEYS:
            out[k] = v
        elif _is_record_list(v):
            if len(v) > MAX_RECORDS:
                head, tail = MAX_RECORDS // 2, MAX_RECORDS - MAX_RECORDS // 2
                kept = [slim(e, _depth + 1) for e in v[:head]]
                kept += [slim(e, _depth + 1) for e in v[-tail:]]
                out[k] = kept
                out[f"{k}_omitted"] = len(v) - MAX_RECORDS
                out[f"{k}_total"] = len(v)
            else:
                out[k] = [slim(e, _depth + 1) for e in v]
        elif isinstance(v, list) and len(v) > MAX_SCALAR_LIST:
            continue
        elif isinstance(v, dict):
            out[k] = slim(v, _depth + 1)
        else:
            out[k] = v
    return out
