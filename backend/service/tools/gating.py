"""Tools that are enabled or disabled by configuration."""

from __future__ import annotations

from backend.common.config import agent as agent_cfg
from backend.common.config import memory as mem_cfg

GATED = {
    "preview_grid_scan": (agent_cfg, "GRID_SCAN"),
    "run_grid_scan": (agent_cfg, "GRID_SCAN"),
    "set_sample_context": (mem_cfg, "RECORD"),
    "query_measurements": (mem_cfg, "QUERY"),
    "annotate_measurement": (mem_cfg, "QUERY"),
    "compare_measurements": (mem_cfg, "QUERY"),
}


def skills_visible() -> bool:
    """Return whether skill guidance is shown to the model."""
    return bool(mem_cfg.REFLECT or mem_cfg.SKILL_INJECT)


def enabled(name: str) -> bool:
    """Return whether a tool is currently exposed and executable."""
    gate = GATED.get(name)
    return gate is None or bool(getattr(*gate))
