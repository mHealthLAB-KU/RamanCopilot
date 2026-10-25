"""On/off switches for each agent memory component."""

from __future__ import annotations

from backend.common.config.env import flag

RECORD = flag("RAMAN_MEM_RECORD", default="1")
SIDE_EFFECT = flag("RAMAN_MEM_SIDE_EFFECT", default="1")
QUERY = flag("RAMAN_MEM_QUERY", default="1")
SUMMARY = flag("RAMAN_MEM_SUMMARY", default="1")
SKILL_INJECT = flag("RAMAN_MEM_SKILL_INJECT", default="1")
REFLECT = flag("RAMAN_MEM_REFLECT", default="1")
RECALL = flag("RAMAN_MEM_RECALL", default="1")

COMPONENTS = ("RECORD", "SIDE_EFFECT", "QUERY", "SUMMARY", "SKILL_INJECT", "REFLECT", "RECALL")
