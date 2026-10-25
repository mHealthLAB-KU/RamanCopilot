"""Settings for running on virtual hardware."""

from __future__ import annotations

from backend.common.config.env import env, flag

VIRTUAL_HW = flag("RAMAN_VIRTUAL_HW", default="0")

VBENCH_SEED = int(env("RAMAN_VBENCH_SEED", default="0"))

def scene_label() -> str | None:
    """Reproduction key of the virtual sample — 'vbench#7'. None for physical hardware (run manifest · startup log)."""
    if not VIRTUAL_HW:
        return None
    return f"vbench#{VBENCH_SEED}"
