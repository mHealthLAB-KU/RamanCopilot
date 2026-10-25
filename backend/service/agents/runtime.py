"""Shared agent runtime: LLM connection, tool dispatch, message handling and sessions."""


from __future__ import annotations

import sys
import time
from typing import Any, Callable, Iterator, Optional

from langchain_core.messages import HumanMessage

from backend.service.agents.logs import run_log
from backend.repo.store import run_store
from backend.service.tools.non_hw_tools.file_tools import FILE_DISPATCH
from backend.common.result import fail, is_ok, normalize, ok
from backend.service.tools.non_hw_tools.data_tools import identify_material
from backend.service.tools.schema import call_with
from backend.service.tools import gating as _gating
from backend.service.memory import recall as _mem_recall
from backend.service.memory import recorder as _mem_recorder
from backend.repo.store.spectrum_store import spectrum_event
from backend.common.config import llm as cfg
from backend.common.config import agent as agent_cfg
from backend.service.agents.tool_slim import slim


_llm_cache: dict[str, object] = {}

_llm_error: str = ""


def _cache_key(tools: Optional[list]) -> str:
    """Return the model cache key (model name and bound tool names)."""
    if tools is None:
        return f"{cfg.OLLAMA_MODEL}|plain"
    names = sorted(t["function"]["name"] for t in tools if isinstance(t, dict) and "function" in t)
    return f"{cfg.OLLAMA_MODEL}|tools:{','.join(names)}"


def get_chat_model(tools: Optional[list] = None):
    """Return the chat model, optionally with tools bound.

    Models are cached by model name and tool set.

    Args:
        tools (list, optional): Tool schemas to bind.

    Returns:
        Runnable or None: The chat model, or None if it could not be created.
    """
    global _llm_error
    key = _cache_key(tools)
    if key in _llm_cache:
        return _llm_cache[key]
    try:
        from backend.repo.llm import providers
        from backend.repo.llm.llm_call import parse as call_parse

        provider, _ = providers.parse_model(cfg.OLLAMA_MODEL)
        _llm_cache[key] = call_parse.build(
            cfg.OLLAMA_MODEL,
            base_url=cfg.OLLAMA_HOST if provider == "ollama" else cfg.API_BASE_URL,
            num_ctx=cfg.NUM_CTX,
            timeout_s=cfg.LLM_TIMEOUT_S,
            max_tokens=cfg.MAX_TOKENS,
            tools=tools,
        )
    except Exception as e:  # noqa: BLE001
        _llm_error = f"{type(e).__name__}: {e}"
        print(f"[llm] could not build the model - {_llm_error}", file=sys.stderr)
        return None
    _llm_error = ""
    return _llm_cache[key]


def llm_unavailable_detail() -> str:
    """Return the message shown when no model is available.

    Returns:
        str: Provider, setting to check and the last error.
    """
    try:
        from backend.repo.llm import providers
        provider, name = providers.parse_model(cfg.OLLAMA_MODEL)
        where = f"{provider}:{name}"
        hint = (f"Ollama server at {cfg.OLLAMA_HOST}" if provider == "ollama"
                else f"{providers.PIP_PACKAGE.get(provider, 'the provider package')} "
                     f"and the API key")
    except Exception:  # noqa: BLE001
        where, hint = cfg.OLLAMA_MODEL, "the configuration"
    return (f"Could not build the LLM ({where}). Check {hint}."
            + (f" Cause: {_llm_error}" if _llm_error else ""))


def get_tool_dispatch():
    """Load the tool dispatch table.

    Returns:
        dict or None: Tool name to function, or None if the hardware modules cannot be imported.
    """
    try:
        from backend.service.tools.tools import TOOL_DISPATCH
        return TOOL_DISPATCH
    except Exception:
        return None


def grid_gate_begin_turn(interactive: bool) -> None:
    """Notify the grid-scan approval gate that a turn starts.

    Args:
        interactive (bool): True for chat sessions, False for unattended runs.
    """
    try:
        from backend.service.tools.hw_tools.acquire_tools import grid_gate_begin_turn as _begin
        _begin(interactive=interactive)
    except Exception:
        pass


RUNTIME_DISPATCH = {
    "identify_material": lambda a: call_with(identify_material, a),
    "set_sample_context": lambda a: call_with(_mem_recorder.set_sample_context, a),
    "query_measurements": lambda a: call_with(_mem_recall.query_measurements, a),
    "annotate_measurement": lambda a: call_with(_mem_recall.annotate_measurement, a),
    "compare_measurements": lambda a: call_with(_mem_recall.compare_measurements, a),
}


_IRRADIATING_TOOLS = {"acquire_spectrum", "run_grid_scan"}


def call_tool(ctx: dict, name: str, args: dict) -> dict:
    """Call one tool and check that its result follows the response format.

    Args:
        ctx (dict): Turn context.
        name (str): Tool name.
        args (dict): Tool arguments.

    Returns:
        dict: Tool result in the ok/fail format.
    """
    return normalize(_dispatch(ctx, name, args), name)


def _batch_block(ctx: dict, name: str) -> Optional[dict]:
    """Return a refusal reason for a second irreversible action in one response, else None."""
    if not agent_cfg.BATCH_GATE:
        return None
    seen = ctx.get("batch_seen")
    if not seen or seen[0] != ctx.get("batch"):
        seen = ctx["batch_seen"] = (ctx.get("batch"), [])
    done = seen[1]

    why = None
    if name in _IRRADIATING_TOOLS and any(n in _IRRADIATING_TOOLS for n in done):
        why = (f"Not executed, and nothing was irradiated. This is not an error and not a safety "
               f"limit: only one irradiation runs per response, and you already called "
               f"{next(n for n in done if n in _IRRADIATING_TOOLS)} in this one, so its result is "
               f"not back yet. Read that result, then call {name} again in your next step. Repeat "
               f"one per step until you have every measurement the task needs.")
    elif name == "run_grid_scan" and "preview_grid_scan" in done:
        why = ("Not executed, and nothing was irradiated. This is not an error and not a safety "
               "limit: preview_grid_scan is in this same response, so you have not seen the "
               "preview image yet. Look at the preview, then call run_grid_scan in your next step.")

    done.append(name)
    return fail(why) if why else None


def _dispatch(ctx: dict, name: str, args: dict) -> dict:
    """Dispatch a tool call and record laser shots in ctx."""
    if not _gating.enabled(name):
        return fail(f"Unknown tool: {name}")
    if name in RUNTIME_DISPATCH:
        return RUNTIME_DISPATCH[name](args)
    if name in FILE_DISPATCH:
        return FILE_DISPATCH[name](args)

    dispatch = ctx["dispatch"]
    if dispatch is None:
        return fail("Hardware is not connected.")
    fn = dispatch.get(name)
    if fn is None:
        return fail(f"Unknown tool: {name}")

    blocked = _batch_block(ctx, name)
    if blocked:
        return blocked

    if name not in _IRRADIATING_TOOLS:
        try:
            return fn(dict(args))
        except Exception as e:
            return fail(f"{type(e).__name__}: {e}")

    try:
        result = fn(dict(args))
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    if is_ok(result):
        if name == "run_grid_scan":
            ctx["laser_shots"].append({"tool": name, "power_mw": result.get("power"),
                                       "exposure_s": result.get("exposure"),
                                       "n_frames": result.get("n_frames"),
                                       "n_points": result.get("n_measured")})
        else:
            a = _mem_recorder._applied(result)
            ctx["laser_shots"].append({"tool": name, "power_mw": a["power_mw"],
                                       "exposure_s": a["exposure_s"], "n_frames": a["n_frames"],
                                       "accumulations": a["num_accumulations"], "kinetic_frames": a["kinetic_count"],
                                       "n_points": 1})
    return result


_RECALL_AFTER = {"move_stage", "move_stage_relative", "move_to_pixel", "acquire_spectrum"}


def _spectrum_image(raw: dict) -> tuple[Optional[str], Optional[str]]:
    """Return the saved spectrum PNG as a small base64 JPEG with a caption, or (None, None)."""
    png = ((raw.get("saved") or {}).get("files") or {}).get("png")
    if not png:
        return None, None
    try:
        import cv2
        from backend.common import vision
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        b64 = vision.encode_for_model(img, max_side=agent_cfg.SPECTRUM_IMAGE_MAX_SIDE,
                                      quality=agent_cfg.SPECTRUM_IMAGE_JPEG_QUALITY)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] spectrum image: {type(e).__name__}: {e}", file=sys.stderr)
        return None, None
    title = (raw.get("saved") or {}).get("title") or "Raman spectrum"
    caption = (f"The spectrum you just measured: {title}.")
    if raw.get("saturated"):
        caption += "The spectrum you just measured is SATURATED."
    return b64, caption


def execute_tool(ctx: dict, name: str, args: dict, tool_call_id: str = "") -> dict:
    """Execute one tool and return its reduced result.

    Image data is removed from the result and returned separately.

    Args:
        ctx (dict): Turn context.
        name (str): Tool name.
        args (dict): Tool arguments.
        tool_call_id (str): Identifier of the tool call.

    Returns:
        dict: name, args, result, img_b64, question, tool_call_id and elapsed_ms.
    """
    ctx["tool_call_order"].append(name)
    t0 = time.time()
    with _mem_recorder.call_scope(ctx, tool_call_id):
        raw = call_tool(ctx, name, args)
        if name in _RECALL_AFTER and raw.get("ok"):
            try:
                hint = _mem_recall.spot_hint()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] memory_hint: {type(e).__name__}: {e}", file=sys.stderr)
                hint = None
            if hint:
                raw["memory_hint"] = hint
    elapsed_ms = (time.time() - t0) * 1000.0
    result = slim(raw)
    img_b64 = result.pop("image_base64", None)
    question = result.pop("question", None)
    if img_b64 is None and name == "acquire_spectrum" and agent_cfg.SPECTRUM_IMAGE and raw.get("ok"):
        img_b64, question = _spectrum_image(raw)
    return {"name": name, "args": args, "result": result, "img_b64": img_b64,
            "question": question, "tool_call_id": tool_call_id, "elapsed_ms": elapsed_ms}


INJECTED_IMAGE = "_injected_image"

HISTORY_MAX_TURNS = 30  # user turns kept in the history


def text_of(msg) -> str:
    """Extract plain text from a message.

    Args:
        msg: LangChain message whose content is a string or a list of blocks.

    Returns:
        str: The text content.
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def image_message(img_b64: str, question: str = "") -> HumanMessage:
    """Build a user message that shows an image to the model.

    Args:
        img_b64 (str): Base64-encoded image.
        question (str): Text sent with the image.

    Returns:
        HumanMessage: Message with a text block and an image block.
    """
    return HumanMessage(
        content=[
            {"type": "text", "text": question or "Microscope camera image:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ],
        additional_kwargs={INJECTED_IMAGE: True},
    )


def trim_history(messages: list) -> list:
    """Build the history for the next turn.

    Old turns are dropped at user-message boundaries and all injected images are removed.

    Args:
        messages (list): Messages of the session.

    Returns:
        list: The trimmed messages.
    """
    def is_user_turn(m) -> bool:
        return isinstance(m, HumanMessage) and not m.additional_kwargs.get(INJECTED_IMAGE, False)

    def is_injected_image(m) -> bool:
        return isinstance(m, HumanMessage) and bool(m.additional_kwargs.get(INJECTED_IMAGE, False))

    user_idx = [i for i, m in enumerate(messages) if is_user_turn(m)]
    if len(user_idx) > HISTORY_MAX_TURNS:
        messages = messages[user_idx[-HISTORY_MAX_TURNS]:]

    return [m for m in messages if not is_injected_image(m)]


def describe_tool(name: str, args: dict, result: dict) -> str:
    """Summarize one tool call as a short line for the chat interface.

    Args:
        name (str): Tool name.
        args (dict): Tool arguments.
        result (dict): Tool result.

    Returns:
        str: Summary line.
    """
    if not isinstance(result, dict):
        return f"🔧 {name} called"
    if not result.get("ok", True):
        err = str(result.get("error", ""))
        if name == "set_sample_context" and "not returned by any web_search" in err:
            return "🔎 Band source not verified — no web search in this session returned those pages yet"
        return f"⚠️ {name} failed: {err}"

    if name == "web_search" and not result.get("results"):
        return "🌐 Web search → no new results"

    if name == "acquire_spectrum":
        return f"📈 Spectrum acquired (max {result.get('max_intensity', 0):.0f} ADU)"
    if name in ("move_stage", "move_stage_relative", "move_to_pixel"):
        pos = result.get("position", {})
        return f"🧭 Moved → ({pos.get('x', '?')}, {pos.get('y', '?')})"
    if name == "analyze_microscope_image":
        return "👁️ Microscope image checked"
    if name == "run_autofocus":
        return "🔬 Autofocus complete"
    if name == "preview_grid_scan":
        return (f"🔲 Grid preview {result.get('rows', '?')}×{result.get('cols', '?')} "
                f"({result.get('n_in_view', '?')}/{result.get('n_points', '?')} in view)")
    if name == "run_grid_scan":
        return (f"🗺️ Grid scan done "
                f"({result.get('n_measured', '?')}/{result.get('n_points', '?')} points)")
    if name == "identify_material":
        hits = result.get("results", [])
        if not hits:
            return "🔬 Material ID → no candidate"
        top = hits[0]
        rest = ", ".join(h["mineral"] for h in hits[1:3])
        return (f"🔬 Material ID → {top['mineral']} {top['score']:.3f}"
                + (f"  (then {rest})" if rest else ""))
    if name == "list_uploaded_files":
        files = result.get("files", [])
        if not files:
            return "📎 Attached files — none"
        return f"📎 Attached files → {', '.join(f.get('filename', '?') for f in files)}"
    if name == "open_file":
        kind = result.get("kind")
        if kind == "image":
            return f"🖼️ Opened image {result.get('file_id', '?')}"
        if kind == "table":
            return (f"🔍 Opened {result.get('filename', '?')} "
                    f"({result.get('n_rows', '?')} rows × {result.get('n_cols', '?')} cols)")
        return f"📄 Opened {result.get('file_id', '?')}"
    if name == "run_analysis":
        return f"🧮 Analysis code executed ({result.get('image_count', 0)} figure(s))"
    return f"🔧 {name} called"


LLM_NOT_CONNECTED = ("LLM is not connected. "
                     "(Check the provider package, the API key, and the host)")

EMPTY_REPLY = ("The model ended its turn without producing an answer or a tool call. "
               "This is usually a malformed tool call that could not be parsed - it is not "
               "something you did wrong. Try asking again.")


def _is_timeout(e: BaseException) -> bool:
    """Return whether an exception is a timeout."""
    text = f"{type(e).__name__} {e}".lower()
    return "timeout" in text or "timed out" in text


def _is_connect_error(e: BaseException) -> bool:
    """Return whether an exception is a connection error."""
    return "connect" in type(e).__name__.lower()


def llm_error_detail(e: Exception, stage: str = "LLM call") -> str:
    """Describe an LLM call exception in one line.

    Args:
        e (Exception): The exception.
        stage (str): Name of the failed step.

    Returns:
        str: Error message. Timeouts are reported separately from other errors.
    """
    if _is_timeout(e):
        return (f"No response from the LLM within {cfg.LLM_TIMEOUT_S:.0f}s ({type(e).__name__}). "
                f"The server may have dropped the request or be overloaded - check that "
                f"{cfg.OLLAMA_HOST} is healthy and retry.")
    return f"{stage} failed: {type(e).__name__}: {e}"


MAX_LLM_RETRIES = 1  # retries for transient errors

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})  # transient server errors

NEVER_RETRY_STATUS = frozenset({400, 401, 403, 404, 405, 422})  # request errors that fail again on retry

RETRY_WAIT_S = 2.0  # seconds before a retry

RETRY_WAIT_MAX_S = 30.0  # give up if Retry-After is longer


def _retry_after_s(e: BaseException) -> Optional[float]:
    """Return the provider's Retry-After value in seconds, or None."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        v = headers.get("retry-after", headers.get("Retry-After"))
        return float(v) if v is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def retry_plan(e: BaseException) -> tuple[bool, Optional[int], float]:
    """Decide whether a failed LLM call should be retried.

    Args:
        e (BaseException): The exception.

    Returns:
        tuple[bool, int or None, float]: Whether to retry, the HTTP status code and the wait time in seconds.
    """
    status = run_log.status_of(e)
    if status in NEVER_RETRY_STATUS:
        return False, status, 0.0
    if status in RETRYABLE_STATUS:
        wait = _retry_after_s(e)
        wait = RETRY_WAIT_S if wait is None else wait
    elif _is_timeout(e) or _is_connect_error(e):
        wait = RETRY_WAIT_S
    else:
        return False, status, 0.0
    return (wait <= RETRY_WAIT_MAX_S), status, wait


def invoke_llm(llm, messages) -> tuple[Any, dict]:
    """Invoke the LLM and retry once on transient errors.

    Args:
        llm: Chat model.
        messages (list): Messages to send.

    Returns:
        tuple[AIMessage, dict]: The response and timing data. Only the successful attempt is counted in call_took_s.
    """
    lost_s, wait_s = 0.0, 0.0
    retried = 0
    errors: list[dict] = []
    codes: list[int] = []

    def tel(call_took_s: float) -> dict:
        return {"call_took_s": round(call_took_s, 4), "retries": retried,
                "retry_codes": codes, "retry_lost_s": round(lost_s, 4),
                "retry_wait_s": round(wait_s, 4), "retry_errors": errors}

    for attempt in range(MAX_LLM_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            return llm.invoke(messages), tel(time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            lost_s += time.perf_counter() - t0
            again, status, wait = retry_plan(e)
            errors.append(run_log.error_info(e))
            if status is not None:
                codes.append(status)
            if not again or attempt >= MAX_LLM_RETRIES:
                e.raman_retry = tel(0.0)  # type: ignore[attr-defined]
                raise
            retried += 1
            time.sleep(wait)
            wait_s += wait
    raise AssertionError("unreachable")


SessionStore = dict


def _used_measurement(ctx: Optional[dict]) -> bool:
    """Return whether the laser was fired in this turn."""
    return bool(set(_IRRADIATING_TOOLS) & set((ctx or {}).get("tool_call_order", [])))


def stream_turn(arch: str, sessions: SessionStore, run: Callable[[list, str], Iterator[dict]],
                user_message: str, session_id: str = "",
                interactive_grid_gate: bool = False) -> Iterator[dict]:
    """Run one agent turn and convert its events into chat events.

    Args:
        arch (str): Architecture name.
        sessions (SessionStore): Session histories.
        run (Callable): Agent loop that takes history and session id.
        user_message (str): User input.
        session_id (str): Session name.
        interactive_grid_gate (bool): Whether the grid-scan approval gate is on.

    Yields:
        dict: Events of type node, chat, done or error.
    """
    sid = session_id or run_store.new_session_id()

    def ev(d: dict) -> dict:
        d["session_id"] = sid
        return d

    run_store.begin_session(sid, arch, isolated=agent_cfg.CHAT_SESSION_ISOLATED)

    try:
        history = sessions.get(sid, [])
        grid_gate_begin_turn(interactive=interactive_grid_gate)

        final_text, final_ctx, final_messages = None, None, history
        for event in run(history, sid):
            etype = event["type"]
            if etype == "tool":
                yield ev({"type": "node", "node": event["name"],
                          "message": describe_tool(event["name"], event["args"], event["result"])})
                sp = spectrum_event(event["result"])
                if sp:
                    yield ev(sp)
            elif etype == "error":
                yield ev({"type": "error", "detail": event["detail"]})
                return
            elif etype == "final":
                final_text, final_ctx = event["text"], event["ctx"]
                final_messages = event["messages"]

        if final_ctx is None:
            yield ev({"type": "error", "detail": "The agent failed to generate a response."})
            return

        sessions[sid] = trim_history(final_messages)
        measured = _used_measurement(final_ctx)
        yield ev({"type": "done", "final_report": final_text} if measured
                 else {"type": "chat", "reply": final_text})

    except Exception as e:
        yield ev({"type": "error", "detail": str(e)})


def run_turn_once(arch: str, run: Callable[[list, str], Iterator[dict]],
                  user_message: str, session_id: str = "") -> dict:
    """Run one agent turn without session history.

    Args:
        arch (str): Architecture name.
        run (Callable): Agent loop that takes history and session id.
        user_message (str): User input.
        session_id (str): Session name. A new one is created if empty.

    Returns:
        dict: Result with final_report.
    """
    sid = session_id or run_store.new_session_id()
    run_store.begin_session(sid, arch, isolated=True)
    grid_gate_begin_turn(interactive=False)

    final_text, final_ctx, error_detail = "", None, None
    for event in run([], sid):
        if event["type"] == "final":
            final_text, final_ctx = event["text"], event["ctx"]
        elif event["type"] == "error":
            error_detail = event["detail"]
            final_text = f"[Error] {event['detail']}"

    return {"final_report": final_text}


__all__ = [
    "get_chat_model", "invoke_llm", "retry_plan", "MAX_LLM_RETRIES",
    "get_tool_dispatch", "grid_gate_begin_turn",
    "identify_material", "call_tool", "execute_tool",
    "INJECTED_IMAGE", "HISTORY_MAX_TURNS", "text_of", "image_message", "trim_history",
    "describe_tool", "LLM_NOT_CONNECTED", "EMPTY_REPLY", "llm_error_detail",
    "llm_unavailable_detail",
    "SessionStore", "stream_turn", "run_turn_once",
]
