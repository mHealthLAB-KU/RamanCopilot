"""Builds the system prompt."""


from __future__ import annotations

import os

from backend.service.agents.prompts import architecture, common


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


AUTONOMOUS = _env_on("RAMAN_AUTONOMOUS")

_PROFILES = {"REACT": architecture.REACT}


def build_system_prompt(arch: str, *, autonomous: bool | None = None) -> str:
    """Assemble the system prompt for an architecture."""
    autonomous = AUTONOMOUS if autonomous is None else autonomous

    profile = _PROFILES.get(arch.strip().upper(), architecture.REACT)
    slots = dict(profile["slots"])

    slots["safety_on_block"] = (common.SAFETY_ON_BLOCK_AUTONOMOUS if autonomous
                                else common.SAFETY_ON_BLOCK_INTERACTIVE)

    blocks = [profile["preamble"]]
    blocks += profile["extra_sections"]
    from backend.service.tools import gating
    rec_on = gating.enabled("set_sample_context")
    cmp_on = gating.enabled("compare_measurements")
    blocks += [common.HARDWARE_RECOVERY,
               common.SAFETY_RULES,
               common.FOCUS_CHECK]
    blocks += [b for b, on in ((common.BAND_SOURCES, rec_on), (common.CONTROLS, rec_on)) if on]
    blocks += [common.MEASUREMENT_PROCESS]
    blocks += [common.IRREVERSIBLE] if (rec_on and cmp_on) else []
    blocks += [
               common.BASIC_EXPERIMENTAL_PRINCIPLES,
               common.SIGNAL_QUALITY,
               common.PREPROCESSING]
    blocks += [b for b in (common.memory_block(), common.skills_block()) if b]

    return "\n\n".join(b.format(**slots) if "{" in b else b for b in blocks) + "\n"


def describe_mode() -> str:
    """Return a one-line description of the prompt mode for the startup log."""
    return f"autonomous={'on' if AUTONOMOUS else 'OFF (ask-the-user gates restored)'}"
