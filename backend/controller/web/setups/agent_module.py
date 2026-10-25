"""Selects the agent module for a request."""


from __future__ import annotations

from typing import Optional

from backend.common.config import agent as cfg

_MODULES = {
    "REACT": "backend.service.agents.architectures.single_agent_ReAct",
}
_DISPLAY = {"REACT": "ReAct"}


def select_agent_module(name: Optional[str] = None):
    """Return (agent module, normalized name) for an architecture name, falling back to the default."""
    import importlib

    key = (name or "").strip().upper()
    if key not in _MODULES:
        key = cfg.AGENT_ARCH.strip().upper()
    if key not in _MODULES:
        key = "REACT"
    return importlib.import_module(_MODULES[key]), _DISPLAY[key]
