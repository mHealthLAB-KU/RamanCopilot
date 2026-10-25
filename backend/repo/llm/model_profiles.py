"""Loads the per-model thinking settings from model_profiles.json."""

from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

PROFILE_PATH = Path(__file__).resolve().parent / "model_profiles.json"

_FALLBACK = {"mode": "off", "note": "not in model_profiles.json - no thinking arguments are sent"}

_cache: Optional[dict] = None
_load_error: str = ""


def load(force: bool = False) -> dict:
    """Load and cache the JSON profiles; never raises (see load_error())."""
    global _cache, _load_error
    if _cache is not None and not force:
        return _cache
    try:
        _cache = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        _load_error = ""
    except Exception as e:  # noqa: BLE001
        _cache = {"profiles": []}
        _load_error = f"{type(e).__name__}: {e}"
    return _cache


def load_error() -> str:
    """Return the reason the JSON failed to load, or an empty string."""
    load()
    return _load_error


def resolve(spec: str) -> dict:
    """Return the thinking profile for a model spec."""
    key = _normalize(spec)
    for p in load().get("profiles", []):
        pat = str(p.get("match", ""))
        if pat and fnmatch(key, pat):
            out = dict(p.get("thinking") or {})
            note = p.get("note", "")
            out["note"] = " ".join(str(x) for x in note) if isinstance(note, list) else str(note)
            out["matched"] = pat
            return out
    return dict(_FALLBACK, matched="<none>")


def thinking_label(spec: str) -> tuple[str, str]:
    """Return (status, reason) describing the thinking setting for log headers."""
    p = resolve(spec)
    mode = p.get("mode", "")
    note = p.get("note", "")
    def _parts(*pairs) -> str:
        return ", ".join(f"{k}={v}" for k, v in pairs if v)

    if mode == "off":
        if p.get("matched") == "<none>":
            return "OFF (no profile)", (
                f"{_normalize(spec)} matches no pattern in model_profiles.json - "
                f"runs without any thinking arguments. Add a line to that file")
        return "OFF", note or "this model has no thinking mode"
    if mode == "on":
        return "ON (think)", note or "sends think=true - supported models only"
    if mode == "builtin":
        extra = ", thoughts" if p.get("include_thoughts") else ""
        return f"ON (builtin{extra})", note or "the model reasons by itself - not switched by us"
    if mode == "adaptive":
        extra = _parts(("effort", p.get("effort")), ("display", p.get("display")))
        return (f"ON (adaptive{', ' + extra if extra else ''})",
                note or "sends thinking={'type':'adaptive'}")
    if mode == "effort":
        extra = _parts(("effort", p.get("effort", "?")), ("summary", p.get("summary")))
        return f"ON ({extra})", note or "sends reasoning_effort"
    if mode == "budget":
        extra = _parts(("budget", p.get("budget_tokens")),
                       ("thoughts", "on" if p.get("include_thoughts") else ""))
        return f"ON (budget{', ' + extra if extra else ''})", note or "sends thinking_budget"
    return "UNKNOWN", f"unknown mode={mode!r} - check model_profiles.json"


def thinking_label_mode_known(mode: str) -> bool:
    """Return whether thinking_label knows this mode."""
    return mode in ("off", "on", "builtin", "adaptive", "effort", "budget")


def reasons_by_default(spec: str) -> bool:
    """Return whether this model reasons regardless of what is sent."""
    return resolve(spec).get("mode") in ("on", "builtin", "adaptive", "effort")


def _normalize(spec: str) -> str:
    """Normalize a spec to '<provider>:<model>' (no prefix means ollama)."""
    raw = str(spec or "").strip()
    head, sep, tail = raw.partition(":")
    if sep and head.strip().lower() in ("ollama", "openai", "anthropic", "google"):
        return f"{head.strip().lower()}:{tail.strip()}"
    return f"ollama:{raw}"


def describe() -> str:
    """Return a one-line summary of the loaded profiles."""
    err = load_error()
    n = len(load().get("profiles", []))
    if err:
        return f"model_profiles.json failed to load ({err}) - all defaults assumed"
    return f"model_profiles.json · {n} profiles"


__all__ = ["PROFILE_PATH", "load", "load_error", "resolve", "thinking_label",
           "thinking_label_mode_known",
           "reasons_by_default", "describe"]
