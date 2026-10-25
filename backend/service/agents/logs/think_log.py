"""Human-readable log of the model reasoning in each turn."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

from backend.repo.store import LOG_ROOT
from backend.repo.store.run_store import current as _session_now, existing_log, log_stem, manifest as _manifest

LOG_DIR = LOG_ROOT

ENABLED = os.environ.get("RAMAN_THINK_LOG", "1").strip().lower() not in ("0", "false", "no")

_SESSION_FILES: dict[tuple[str, str], Path] = {}

_LOCK = threading.Lock()

_W = 79

_TURN_MARK = "━━━ TURN "
_END_MARK = "━━━ END "

_RESULT_PREVIEW = 200  # longer results are written as indented JSON

_PAD = " " * 7

_ARTIFACTS_INLINE = 4  # artifacts listed per tool call


def _cols(text: str) -> int:
    """Return the terminal display width of a string (wide characters count as two)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _bar(text: str, fill: str = "━") -> str:
    """Pad text with fill characters to the log width."""
    return text + fill * max(0, _W - _cols(text))


def _brief(obj: Any, limit: int = _RESULT_PREVIEW) -> str:
    """Render a dict or value on one line, truncating if long."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        s = str(obj)
    s = " ".join(s.split())
    return s if len(s) <= limit else f"{s[:limit]}... (+{len(s) - limit} chars)"


def _is_block(value: Any) -> bool:
    """Return whether an argument is a multi-line string that should be printed as a block."""
    return isinstance(value, str) and "\n" in value


def _block_lines(value: str) -> int:
    """Return the number of lines of a block, excluding leading and trailing blank lines."""
    return value.strip("\n").count("\n") + 1


def _split_blocks(obj: Any) -> tuple[Any, list[tuple[str, str]]]:
    """Split multi-line string fields out of a dict as (rest, [(key, text), ...])."""
    if not isinstance(obj, dict):
        return obj, []
    blocks = [(k, v) for k, v in obj.items() if _is_block(v)]
    if not blocks:
        return obj, []
    return {k: v for k, v in obj.items() if not _is_block(v)}, blocks


def _result(obj: Any) -> str:
    """Render a tool result for the log without truncation."""
    one = _brief(obj, sys.maxsize)
    if len(one) <= _RESULT_PREVIEW:
        return one
    try:
        body = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:  # noqa: BLE001
        return one
    return "\n" + "\n".join(_PAD + ln for ln in body.splitlines())


def _kwargs(args: dict) -> str:
    """Render tool arguments as name=value, like a call expression."""
    return ", ".join(
        f"{k}=<{_block_lines(v)} lines>" if _is_block(v) else f"{k}={_brief(v, 60)}"
        for k, v in (args or {}).items())


def _model_header() -> tuple[str, str]:
    """Return (model spec, thinking setting label) for the log header."""
    try:
        from backend.common.config.llm import OLLAMA_MODEL
        from backend.repo.llm.model_profiles import thinking_label
        state, _why = thinking_label(OLLAMA_MODEL)
        return OLLAMA_MODEL, state
    except Exception as e:  # noqa: BLE001
        return "", f"(config unreadable: {type(e).__name__})"


def _thinking_expected() -> bool:
    """Return whether this model is expected to produce thinking."""
    try:
        from backend.common.config.llm import OLLAMA_MODEL
        from backend.repo.llm.model_profiles import reasons_by_default
        return reasons_by_default(OLLAMA_MODEL)
    except Exception:  # noqa: BLE001
        return True


class ThinkLog:
    """Human-readable thinking log of one turn (same method names as RunLog)."""

    def __init__(self, agent: str, session_id: str, question: str):
        self.agent = str(agent or "")
        self.session_id = str(session_id or "")
        self.question = str(question or "")
        self._t0 = time.perf_counter()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._step = 0
        self._n_tools = 0
        self._closed = False
        self._path: Optional[Path] = None
        self._thinking_expected = _thinking_expected()
        self._seen_artifacts: set[str] = set()
        self._turn_artifacts: list[dict] = []
        if ENABLED:
            self._open()

    def _open(self) -> None:
        """Choose the file and write the header or a turn separator."""
        key = (self.agent, self.session_id)
        path = _SESSION_FILES.get(key)
        if path is None:
            digits = "".join(c for c in self.started_at if c.isdigit())
            stamp = f"{digits[:8]}_{digits[8:14]}" if len(digits) >= 14 else digits
            path = (existing_log(LOG_DIR, self.agent, self.session_id, ".log")
                    or LOG_DIR / f"{log_stem(self.agent, stamp, self.session_id)}.log")
            _SESSION_FILES[key] = path
        self._path = path

        if not path.exists():
            model, thinking = _model_header()
            head = "  ·  ".join(x for x in (self.agent, model, f"thinking {thinking}") if x)
            self._write("\n".join([
                "═" * _W,
                f" {head}",
                f" session  {self.session_id}",
                f" started  {self.started_at}",
                "═" * _W,
                "",
            ]))
        self._write("\n".join([
            "",
            _bar(f"{_TURN_MARK}{self._turn_no()} "),
            "",
            "[QUESTION]",
            self.question,
            "",
        ]))

    def _turn_no(self) -> int:
        """Return the turn number by counting turn headers in the file."""
        try:
            return self._path.read_text(encoding="utf-8").count("\n" + _TURN_MARK) + 1
        except Exception:  # noqa: BLE001
            return 1

    def _write(self, text: str) -> None:
        """Write to disk immediately."""
        if not ENABLED or self._path is None:
            return
        try:
            with _LOCK:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(text if text.endswith("\n") else text + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[think_log] write failed: {type(e).__name__}: {e}", file=sys.stderr)

    def _artifacts_new(self) -> list[dict]:
        """Return artifacts from the session manifest not yet written to the log."""
        try:
            if _session_now().get("session_id") != self.session_id:
                return []
            arts = _manifest().get("artifacts") or []
        except Exception:  # noqa: BLE001
            return []
        new = [a for a in arts if a.get("path") not in self._seen_artifacts]
        self._seen_artifacts.update(a.get("path") for a in new)
        self._turn_artifacts.extend(new)
        return new

    def _blocks(self, pairs: list[tuple[str, str]]) -> None:
        """Write multi-line text verbatim (no truncation, no indentation)."""
        for k, v in pairs:
            self._write(f"[{k.upper()}]\n{v.strip(chr(10))}\n\n")

    def _artifact_lines(self) -> None:
        """Write newly created artifacts below the tool line."""
        new = self._artifacts_new()
        for a in new[:_ARTIFACTS_INLINE]:
            self._write(f"       + {str(a.get('kind') or '?'):<11} {a.get('path') or ''}")
        if len(new) > _ARTIFACTS_INLINE:
            self._write(f"       + {len(new) - _ARTIFACTS_INLINE} more files")

    def _retry_note(self, retry: dict) -> None:
        """Write one line per failed retry attempt."""
        if not retry.get("retries"):
            return
        for i, err in enumerate(retry.get("retry_errors") or [], start=1):
            code = f"({err.get('status')})" if err.get("status") else "(no status code)"
            self._write(f"[RETRY] attempt {i} failed - {err.get('type')}{code} - "
                        f"{_brief(err.get('message'), 300)}")
        self._write(f"        burned {retry.get('retry_lost_s', 0.0):.2f}s, waited "
                    f"{retry.get('retry_wait_s', 0.0):.1f}s, then resent\n")

    def llm(self, res: dict, call_took_s: float, **retry: Any) -> None:
        """Record one LLM response."""
        self._step += 1
        think = res.get("thinking_text") or ""
        text = res.get("text") or ""
        tokens = int(res.get("thinking_tokens") or 0)

        bits = [f"{float(call_took_s):.2f}s",
                f"in {int(res.get('input_tokens') or 0)}",
                f"out {int(res.get('output_tokens') or 0)}"]
        if think:
            bits.append(f"thinking {len(think):,} ch")
        if tokens:
            bits.append(f"{tokens:,} tok")
        if res.get("is_blocked"):
            bits.append(f"BLOCKED({res.get('stop_reason') or '?'})")
        elif res.get("is_truncated"):
            bits.append(f"TRUNCATED({res.get('stop_reason') or '?'})")
        if retry.get("retries"):
            bits.append(f"{retry['retries']} retry")
        self._rule(f"step {self._step}", " · ".join(bits))
        self._retry_note(retry)

        if think:
            self._write(f"[THINKING]\n{think}\n")
        elif tokens:
            self._write(f"[THINKING]  no transcript - token count only ({tokens:,} tok). "
                        f"openai/google return the text only when summaries are enabled.\n")
        elif self._thinking_expected:
            self._write("[THINKING]  none - thinking was enabled but nothing came back. "
                        "Whether the model withheld it or did not think cannot be told "
                        "from this log (see the thinking setting in the header).\n")

        if text.strip():
            self._write(f"\n[TEXT]\n{text}\n")

    def llm_failed(self, exc: BaseException, call_took_s: float = 0.0,
                   **retry: Any) -> None:
        """Record an LLM call that failed after all retries."""
        self._step += 1
        burned = float(retry.get("retry_lost_s") or call_took_s or 0.0)
        tail = f"{burned:.2f}s burned · no response"
        if retry.get("retries"):
            tail += f" · {retry['retries']} retry exhausted"
        self._rule(f"step {self._step}", tail)
        self._retry_note(retry)
        self._write(f"[FAILED] {type(exc).__name__}: {exc}\n")

    def tool(self, name: str, args: dict, ex: dict) -> None:
        """Record one tool execution."""
        self._n_tools += 1
        result = ex.get("result")
        ok = result.get("ok", True) if isinstance(result, dict) else True
        rest, out_blocks = _split_blocks(result)
        self._write(f"\n[TOOL] {name}({_kwargs(args)})")
        self._blocks(_split_blocks(args)[1])
        line = _result(rest)
        self._write(f"{_PAD}{'OK' if ok else 'FAILED'} · "
                    f"{float(ex.get('elapsed_ms') or 0.0):.0f}ms"
                    + (line if line.startswith("\n") else f" · {line}"))
        self._blocks(out_blocks)
        self._artifact_lines()

    def tool_failed(self, name: str, args: dict, exc: BaseException,
                    took_ms: float = 0.0) -> None:
        """Record a tool that raised an exception."""
        self._n_tools += 1
        self._write(f"\n[TOOL] {name}({_kwargs(args)})")
        self._blocks(_split_blocks(args)[1])
        self._write(f"{_PAD}RAISED · {float(took_ms):.0f}ms · {type(exc).__name__}: {exc}")
        self._artifact_lines()

    def close(self, answer: str = "", ctx: Optional[dict] = None,
              error: Any = None, response_type: str = "done") -> None:
        """Close the turn (idempotent)."""
        if self._closed:
            return
        self._closed = True
        ctx = ctx or {}
        shots = len(ctx.get("laser_shots") or [])
        bits = [f"{time.perf_counter() - self._t0:.1f}s",
                f"{self._step} LLM calls", f"{self._n_tools} tool calls"]
        if shots:
            bits.append(f"{shots} laser shots")
        if response_type != "done":
            bits.append(response_type)
        self._artifacts_new()
        if self._turn_artifacts:
            self._write("\n" + _bar(f"━━━ ARTIFACTS · {len(self._turn_artifacts)} "))
            for a in self._turn_artifacts:
                self._write(f" {str(a.get('saved_at') or ''):<9}"
                            f"{str(a.get('kind') or '?'):<12} {a.get('path') or ''}")
        self._write("\n" + _bar(f"{_END_MARK}· {' · '.join(bits)} "))
        if error:
            self._write(f"\n[ERROR]\n{error if isinstance(error, str) else _brief(error, 1500)}\n")
        if answer.strip():
            self._write(f"\n[FINAL ANSWER]\n{answer}\n")
        refl = ctx.get("reflection")
        if refl:
            self._write("\n" + _bar(f"━━━ REFLECTION · version={refl.get('version')} · {refl.get('reason')} "))
            for i, call in enumerate(refl.get("io") or [], 1):
                self._write(f"\n[REFLECTION INPUT {i}]\n{call.get('user')}\n")
                self._write(f"\n[REFLECTION OUTPUT {i}]\n{call.get('output')}\n")
            for d in refl.get("dropped") or []:
                self._write(f"[DROPPED] {d.get('why')} :: {(d.get('rule') or {}).get('rule')}")

    def fail(self, detail: Any, ctx: Optional[dict] = None) -> None:
        self.close(answer="", ctx=ctx, error=detail, response_type="error")

    def _rule(self, label: str, tail: str = "") -> None:
        """Return the step separator line."""
        left = f"─── {label} "
        pad = max(3, _W - _cols(left) - _cols(tail) - 1)
        self._write("\n" + left + "─" * pad + (f" {tail}" if tail else "") + "\n")
