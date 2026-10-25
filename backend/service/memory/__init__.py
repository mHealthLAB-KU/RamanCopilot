"""Agent memory: measurement records, recall and reflection."""

from __future__ import annotations

import sys
import uuid
from typing import Callable

from backend.common.config import memory as mem_cfg


def begin_run(ctx: dict) -> str:
    """Start a run and fix the memory context for its duration.

    Issues a run id and, if enabled, fixes the skill version and the record summary at their current values.

    Args:
        ctx (dict): Turn context.

    Returns:
        str: Text appended to the system prompt. Empty if nothing is enabled.
    """
    ctx["mem_run_id"] = str(uuid.uuid4())
    ctx["mem_skill_version"] = None
    blocks = []
    try:
        if mem_cfg.SKILL_INJECT:
            from backend.repo.store import skill_store
            v = skill_store.latest_version()
            if v:
                ctx["mem_skill_version"] = v
        if mem_cfg.SUMMARY:
            from backend.service.memory.recall import summary_text
            from backend.service.memory.recorder import sample_ctx
            text = summary_text(sample_ctx().get("sample_name"))
            if text:
                blocks.append(f"[Past measurements]\n{text}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] memory.begin_run: {type(e).__name__}: {e}", file=sys.stderr)
    return ("\n\n" + "\n\n".join(blocks) + "\n") if blocks else ""


def end_run(ctx: dict, ask: Callable[[str, str], str]) -> dict | None:
    """Finish a run and run reflection if enabled.

    Args:
        ctx (dict): Turn context.
        ask (Callable[[str, str], str]): Function that sends a system and user prompt to the LLM.

    Returns:
        dict or None: Reflection result, or None if reflection is off.
    """
    if not mem_cfg.REFLECT or not ctx.get("mem_run_id"):
        return None
    io: list[dict] = []

    def logged(system: str, user: str) -> str:
        out = ask(system, user)
        io.append({"system": system, "user": user, "output": out})
        return out
    try:
        from backend.common.config import llm as llm_cfg
        from backend.service.memory import reflection
        res = reflection.reflect(ctx["mem_run_id"], logged, model=llm_cfg.OLLAMA_MODEL)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] memory.end_run: {type(e).__name__}: {e}", file=sys.stderr)
        res = {"version": None, "reason": f"error: {type(e).__name__}: {e}", "dropped": []}
    res = dict(res or {})
    res["io"] = io
    res["dropped"] = [{"rule": d[0], "why": d[1]} if isinstance(d, tuple) else d for d in res.get("dropped") or []]
    ctx["reflection"] = res
    return res
