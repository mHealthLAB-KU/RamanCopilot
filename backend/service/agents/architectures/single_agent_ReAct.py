"""Single ReAct agent that controls the Raman microscope with one LLM and all tools."""


from __future__ import annotations

import json
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.service.agents.logs import run_log, think_log
from backend.service.agents import runtime
from backend.service.memory import provenance as _provenance
from backend.service import memory
from backend.repo.llm.llm_response import parse as response_parse
from backend.repo.store import run_store
from backend.service.agents.prompts import AUTONOMOUS, build_system_prompt
from backend.service.tools.tools import ALL_TOOLS

ARCH = "ReAct"
SYSTEM_PROMPT = build_system_prompt("ReAct")

MAX_AGENT_STEPS = 100  # LLM calls per turn

SOFT_BUDGET_LEFT = 10  # steps left when the model is asked to finish


def _get_llm():
    """Return the chat model with tools bound, or None."""
    from backend.service.tools.tools import agent_tools
    return runtime.get_chat_model(agent_tools())


def _without_thinking(llm):
    """Return a copy of the model with thinking disabled (unchanged if unknown)."""
    try:
        if getattr(llm, "thinking", None):
            return llm.model_copy(update={"thinking": {"type": "disabled"}})
        if getattr(llm, "reasoning", None):
            return llm.model_copy(update={"reasoning": False})
    except Exception:  # noqa: BLE001
        pass
    return llm


def _ask_plain(system: str, user: str) -> str:
    """Call the LLM once without tools and without thinking (used by reflection)."""
    llm = runtime.get_chat_model(None)
    if llm is None:
        raise RuntimeError(runtime.llm_unavailable_detail())
    llm = _without_thinking(llm)
    ai_msg, _ = runtime.invoke_llm(llm, [SystemMessage(content=system), HumanMessage(content=user)])
    return response_parse.parse(ai_msg)["text"]


def _budget_note(step: int) -> str:
    """Return the step-budget line appended to the system prompt each step."""
    left = MAX_AGENT_STEPS - step - 1
    note = (f"\n\n[Budget]\nThis is step {step + 1} of {MAX_AGENT_STEPS}; "
            f"{left} step(s) remain in this turn.")
    if left <= SOFT_BUDGET_LEFT:
        note += (" You are almost out of steps. Do not start new measurements - write your final "
                 "report now from what you already have, and state plainly what you did not get to.")
    return note


def run_stream(llm, history: list, user_message: str, session_id: str = "") -> Iterator[dict]:
    """Run the ReAct loop until the model stops calling tools.

    Each step sends the history to the model, runs all requested tools and adds their results.

    Args:
        llm: Chat model with the tools bound.
        history (list): Messages from earlier turns.
        user_message (str): User input of this turn.
        session_id (str): Session name used for logs.

    Yields:
        dict: Events of type tool, error or final.
    """
    if llm is None:
        yield {"type": "error", "detail": runtime.llm_unavailable_detail()}
        return

    ctx = {"dispatch": runtime.get_tool_dispatch(), "laser_shots": [], "tool_call_order": []}
    mem_note = memory.begin_run(ctx)
    messages: list[BaseMessage] = list(history) + [HumanMessage(content=user_message)]
    _provenance.note_user_message(user_message)

    run = run_log.RunLog(ARCH, session_id, user_message)
    think = think_log.ThinkLog(ARCH, session_id, user_message)

    try:
        for step in range(MAX_AGENT_STEPS):
            try:
                session_note = run_store.summary_for_prompt()
                system_text = (SYSTEM_PROMPT
                               + (f"\n\n[This session]\n{session_note}\n" if session_note else "")
                               + mem_note
                               + _budget_note(step))
                ai_msg, tel = runtime.invoke_llm(
                    llm, [SystemMessage(content=system_text)] + messages)
                res = response_parse.parse(ai_msg)
                run.llm(res, **tel)
                think.llm(res, **tel)
            except Exception as e:
                tel = getattr(e, "raman_retry", {})
                run.llm_failed(e, **tel)
                think.llm_failed(e, **tel)
                run.fail(e, ctx)
                think.fail(e, ctx)
                yield {"type": "error", "detail": runtime.llm_error_detail(e, "LLM call")}
                return

            for st in res["server_tools"]:
                _provenance.note_web_result(st["result"])
                ex = {"result": st["result"], "elapsed_ms": 0.0}
                run.tool(st["name"], st["args"], ex)
                think.tool(st["name"], st["args"], ex)
                yield {"type": "tool", "name": st["name"], "args": st["args"],
                       "result": st["result"], "ms": 0.0}

            if not res["tool_calls"] and res["stop_reason"] == "pause_turn":
                messages.append(AIMessage(content=res["history"]))
                continue

            if not res["tool_calls"]:
                final_text = res["text"].strip() or runtime.EMPTY_REPLY
                messages.append(AIMessage(content=res["history"]))
                memory.end_run(ctx, _ask_plain)
                run.close(answer=final_text, ctx=ctx)
                think.close(answer=final_text, ctx=ctx)
                yield {"type": "final", "text": final_text, "ctx": ctx, "messages": messages}
                return

            messages.append(AIMessage(content=res["history"], tool_calls=res["tool_calls"]))

            ctx["batch"] = step
            pending_images: list[BaseMessage] = []
            for tc in res["tool_calls"]:
                name = tc["name"]
                args = dict(tc.get("args") or {})

                try:
                    ex = runtime.execute_tool(ctx, name, args, tc.get("id") or "")
                except Exception as e:
                    run.tool_failed(name, args, e)
                    think.tool_failed(name, args, e)
                    raise
                run.tool(name, args, ex)
                think.tool(name, args, ex)
                if name == "web_search":
                    _provenance.note_web_result(ex["result"])
                yield {"type": "tool", "name": name, "args": args, "result": ex["result"],
                       "ms": round(ex["elapsed_ms"], 1)}

                messages.append(ToolMessage(
                    content=json.dumps(ex["result"], ensure_ascii=False, default=str),
                    tool_call_id=ex["tool_call_id"],
                ))
                if ex["img_b64"]:
                    pending_images.append(runtime.image_message(ex["img_b64"], ex["question"]))

            messages.extend(pending_images)

        capped = (f"Stopped after reaching the maximum number of steps ({MAX_AGENT_STEPS}). "
                  f"Please check the progress and request again.")
        run.close(answer=capped, ctx=ctx, response_type="capped")
        think.close(answer=capped, ctx=ctx, response_type="capped")
        yield {"type": "final", "text": capped, "ctx": ctx, "messages": messages}
    except Exception as e:
        run.fail(e, ctx)
        think.fail(e, ctx)
        raise


_SESSIONS: runtime.SessionStore = {}


def stream_experiment(user_message: str, session_id: str = "") -> Iterator[dict]:
    """Run the agent and yield events for the web interface.

    Args:
        user_message (str): User input.
        session_id (str): Session name. Earlier turns of this session are used as history.

    Yields:
        dict: Server-Sent Events for the chat interface.
    """
    return runtime.stream_turn(
        ARCH, _SESSIONS,
        lambda history, sid: run_stream(_get_llm(), history, user_message, sid),
        user_message, session_id,
        interactive_grid_gate=not AUTONOMOUS)


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """Run the agent once without session history.

    Args:
        user_message (str): User input.
        session_id (str): Session name for logs and saved files. A new one is created if empty.

    Returns:
        dict: Result with final_report.
    """
    return runtime.run_turn_once(
        ARCH,
        lambda history, sid: run_stream(_get_llm(), history, user_message, sid),
        user_message, session_id)
