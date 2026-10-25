
from __future__ import annotations

import collections
import re

from backend.agents.logs.reason_log import TRUNCATED_REASONS  # noqa: E402


def _sum(rows, key) -> float:
    return round(sum(float(r.get(key) or 0.0) for r in rows), 4)


def _sum_thinking(llm_calls: list) -> str:
    """Per-call "<n><unit>" strings -> one total, unit preserved. Empty when nothing thought."""
    total, unit = 0, ""
    for c in llm_calls:
        m = re.match(r"(\d+)(ch|tok)$", str(c.get("thinking") or ""))
        if m:
            total += int(m.group(1))
            unit = unit or m.group(2)
    return f"{total}{unit}" if total else ""


def _tps(outs: list, timing: dict, llm_calls: list) -> tuple:
    """Return (tokens/s, basis) where basis is "eval" (Ollama) or "wall" (API)."""
    total_out = sum(outs)
    if not total_out:
        return None, ""
    if timing["llm_eval_s"]:
        return round(total_out / timing["llm_eval_s"], 2), "eval"
    wall = _sum(llm_calls, "wall_s")
    if wall:
        return round(total_out / wall, 2), "wall"
    return None, ""


def build(*, item: dict, sid: str, wall_s: float, llm_calls: list[dict], turn: dict,
          answer: str, message: str, artifacts: list[dict], setup: dict,
          restore_failed: list, log_paths: dict, config: dict,
          before: dict | None = None, after: dict | None = None,
          new_files: list | None = None) -> dict:
    """Assemble the full per-item record. Never raises — a missing piece becomes null."""
    tools = list(turn.get("Tool calls") or [])
    tool_ms = round(sum(float(c.get("ms") or 0.0) for c in tools), 1)

    first = llm_calls[0] if llm_calls else {}
    ttft = ((first.get("load_s") or 0.0) + (first.get("prompt_eval_s") or 0.0)) or None
    agent_s = float(turn.get("total_duration_s") or 0.0)
    llm_total = _sum(llm_calls, "total_s") or _sum(llm_calls, "wall_s")
    timing = {
        "wall_s": round(wall_s, 3),
        "agent_s": agent_s,
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "llm_total_s": llm_total,
        "llm_eval_s": _sum(llm_calls, "eval_s"),
        "llm_prompt_eval_s": _sum(llm_calls, "prompt_eval_s"),
        "llm_load_s": _sum(llm_calls, "load_s"),
        "tool_s": round(tool_ms / 1000.0, 3),
        "retry_s": round(_sum(llm_calls, "retry_lost_s") + _sum(llm_calls, "retry_wait_s"), 3),
        "restore_s": 0.0,
        "overhead_s": round(max(0.0, wall_s - agent_s), 3),
    }

    prompts = [int(c.get("prompt_tokens") or 0) for c in llm_calls]
    outs = [int(c.get("output_tokens") or 0) for c in llm_calls]
    num_ctx = int(config.get("num_ctx") or 0)
    llm = {
        "n_calls": len(llm_calls),
        "n_retries": sum(int(c.get("retries") or 0) for c in llm_calls),
        "retry_codes": sorted({int(s) for c in llm_calls for s in (c.get("retry_codes") or [])}),
        "prompt_tokens": sum(prompts),
        "output_tokens": sum(outs),
        "max_prompt_tokens": max(prompts, default=0),
        "ctx_utilization": round(max(prompts, default=0) / num_ctx, 4) if num_ctx else None,
        "tok_per_s": _tps(outs, timing, llm_calls)[0],
        "tok_per_s_basis": _tps(outs, timing, llm_calls)[1],
        "thinking": _sum_thinking(llm_calls),
        "text_chars": sum(int(c.get("text_chars") or 0) for c in llm_calls),
        "truncated_calls": sum(1 for c in llm_calls
                               if str(c.get("done_reason") or "") in TRUNCATED_REASONS),
        "calls": llm_calls,
    }

    by_name: dict = collections.OrderedDict()
    for c in tools:
        e = by_name.setdefault(c.get("name"), {"n": 0, "ms": 0.0, "n_failed": 0})
        e["n"] += 1
        e["ms"] = round(e["ms"] + float(c.get("ms") or 0.0), 1)
        e["n_failed"] += 0 if c.get("ok", True) else 1
    tool_rec = {
        "n_calls": len(tools),
        "n_failed": sum(1 for c in tools if not c.get("ok", True)),
        "order": list(turn.get("tool_call_order") or []),
        "by_name": by_name,
        "calls": [{k: c.get(k) for k in ("step", "name", "args", "ok", "error", "ms")}
                  for c in tools],
    }

    kinds = collections.Counter(a.get("kind") for a in artifacts)
    return {
        "item": item["id"],
        "axis": item["axis_key"],
        "axis_name": AXIS_EN.get(item["axis_key"], item["axis_key"]),
        "points": item["points"],
        "sid": sid,
        "group": item["group"],
        "prompt_mode": item["prompt_mode"],
        "timing": timing,
        "llm": llm,
        "tools": tool_rec,
        "safety": {"dose_mj": float(turn.get("dose_mj") or 0.0)},
        "artifacts": {
            "n_measurements": kinds.get("measurement", 0),
            "n_saved_spectra": kinds.get("spectra", 0),
            "n_figures": kinds.get("figure", 0),
            "n_points": kinds.get("points", 0),
            "paths": [a.get("path") for a in artifacts],
        },
        "io": {
            "question": item["question_en"],
            "message": message,
            "message_chars": len(message),
            "answer": answer,
            "answer_chars": len(answer),
        },
        "setup": setup,
        "state": {"before": before or {}, "after": after or {}},
        "new_files": sorted(new_files or []),
        "restore_failed": list(restore_failed or []),
        "turn_error": turn.get("error"),
        "logs": log_paths,
        "config": config,
    }


AXIS_EN = {
    "S": "single-action",
    "C": "composite-action",
    "P": "preprocessing-peak-analysis",
    "A": "acquisition-analysis",
    "X": "safety-exception",
    "D": "diagnosis-response",
    "L": "library-matching",
}

CSV_COLUMNS = [
    "item", "axis", "axis_name", "points", "earned", "verdict",
    "n_pass", "n_total",
    "wall_s", "agent_s", "ttft_s", "llm_total_s", "llm_eval_s", "llm_prompt_eval_s",
    "tool_s", "restore_s", "overhead_s",
    "llm_calls", "prompt_tokens", "output_tokens", "max_prompt_tokens",
    "ctx_utilization", "tok_per_s", "tok_per_s_basis", "thinking", "truncated_calls",
    "tool_calls", "tool_failed",
    "dose_mj", "n_measurements", "n_saved_spectra", "n_figures", "n_points",
    "answer_chars", "setup_ok", "restore_failed", "turn_error", "sid",
]


def csv_row(rec: dict) -> list:
    """One flat row for scores.csv, in CSV_COLUMNS order."""
    t, l, tl = rec["timing"], rec["llm"], rec["tools"]
    a, io = rec["artifacts"], rec["io"]
    return [
        rec["item"], rec["axis"], rec["axis_name"], rec["points"],
        "" if rec.get("earned") is None else rec["earned"], rec.get("verdict", ""),
        rec.get("n_pass", ""), rec.get("n_total", ""),
        t["wall_s"], t["agent_s"], t["ttft_s"], t["llm_total_s"], t["llm_eval_s"],
        t["llm_prompt_eval_s"], t["tool_s"], t["restore_s"], t["overhead_s"],
        l["n_calls"], l["prompt_tokens"], l["output_tokens"], l["max_prompt_tokens"],
        l["ctx_utilization"], l["tok_per_s"], l["tok_per_s_basis"], l["thinking"],
        l["truncated_calls"],
        tl["n_calls"], tl["n_failed"],
        rec["safety"]["dose_mj"], a["n_measurements"], a["n_saved_spectra"],
        a["n_figures"], a["n_points"],
        io["answer_chars"], (rec.get("setup") or {}).get("ok", ""),
        ";".join(rec.get("restore_failed") or []), rec.get("turn_error") or "",
        rec["sid"],
    ]


if __name__ == "__main__":
    calls = [
        {"step": 1, "wall_s": 12.0, "prompt_tokens": 9000, "output_tokens": 300,
         "load_s": 1.5, "prompt_eval_s": 0.8, "eval_s": 9.0, "total_s": 11.5,
         "done_reason": "stop", "thinking": "400ch", "text_chars": 200, "n_tool_calls": 1},
        {"step": 2, "wall_s": 20.0, "prompt_tokens": 12000, "output_tokens": 700,
         "load_s": 0.0, "prompt_eval_s": 1.2, "eval_s": 18.0, "total_s": 19.4,
         "done_reason": "length", "thinking": "900ch", "text_chars": 800, "n_tool_calls": 0},
    ]
    turn = {"total_duration_s": 40.0, "dose_mj": 0.19,
            "tool_call_order": ["get_ccd_info", "acquire_spectrum"],
            "Tool calls": [
                {"step": 0, "name": "get_ccd_info", "args": {}, "ok": True, "ms": 120.0},
                {"step": 1, "name": "acquire_spectrum", "args": {"power": 5},
                 "ok": False, "error": "boom", "ms": 3000.0}]}
    item = {"id": "A01", "axis_key": "A", "points": 2, "group": "auto",
            "prompt_mode": "autonomous", "question_en": "Measure once."}

    r = build(item=item, sid="s", wall_s=42.5, llm_calls=calls, turn=turn,
              answer="Max intensity 4411 ADU.", message="Measure once.",
              artifacts=[{"kind": "measurement", "path": "results/x.csv"}],
              setup={"ok": True, "ran": False}, restore_failed=[],
              log_paths={"detail": "d.json"}, config={"num_ctx": 100000})

    assert r["timing"]["ttft_s"] == 2.3, r["timing"]["ttft_s"]
    assert r["llm"]["tok_per_s_basis"] == "eval", "with eval_duration, generation time is the basis"
    assert r["timing"]["llm_eval_s"] == 27.0
    assert r["timing"]["llm_total_s"] == 30.9
    assert r["timing"]["tool_s"] == 3.12, r["timing"]["tool_s"]
    assert r["timing"]["overhead_s"] == 2.5, "runner wall minus agent turn"
    assert r["timing"]["restore_s"] == 0.0

    assert r["llm"]["n_calls"] == 2
    assert r["llm"]["prompt_tokens"] == 21000 and r["llm"]["output_tokens"] == 1000
    assert r["llm"]["max_prompt_tokens"] == 12000
    assert r["llm"]["ctx_utilization"] == 0.12, "biggest prompt vs num_ctx"
    assert r["llm"]["truncated_calls"] == 1, "done_reason == 'length' must be counted"

    for spelling in ("length", "max_tokens", "MAX_TOKENS"):
        one = [{"step": 1, "wall_s": 1.0, "prompt_tokens": 10, "output_tokens": 5,
                "done_reason": spelling, "thinking": "", "text_chars": 5, "n_tool_calls": 0}]
        got = build(item=item, sid="s", wall_s=1.0, llm_calls=one, turn={}, answer="a",
                    message="m", artifacts=[], setup={}, restore_failed=[], log_paths={},
                    config={})["llm"]["truncated_calls"]
        assert got == 1, f"{spelling!r} not counted as truncation"
    for ok_reason in ("stop", "end_turn", "tool_use", "STOP", None):
        one = [{"step": 1, "wall_s": 1.0, "prompt_tokens": 10, "output_tokens": 5,
                "done_reason": ok_reason, "thinking": "", "text_chars": 5, "n_tool_calls": 0}]
        got = build(item=item, sid="s", wall_s=1.0, llm_calls=one, turn={}, answer="a",
                    message="m", artifacts=[], setup={}, restore_failed=[], log_paths={},
                    config={})["llm"]["truncated_calls"]
        assert got == 0, f"{ok_reason!r} counted as truncation"
    assert abs(r["llm"]["tok_per_s"] - 1000 / 27.0) < 0.01

    assert r["tools"]["n_calls"] == 2 and r["tools"]["n_failed"] == 1
    assert r["tools"]["by_name"]["acquire_spectrum"] == {"n": 1, "ms": 3000.0, "n_failed": 1}
    assert r["axis_name"] == "acquisition-analysis"
    assert r["safety"]["dose_mj"] == 0.19
    assert r["artifacts"]["n_measurements"] == 1
    assert r["state"] == {"before": {}, "after": {}} and r["new_files"] == []
    r2 = build(item=item, sid="s", wall_s=1.0, llm_calls=[], turn={}, answer="", message="",
               artifacts=[], setup={}, restore_failed=[], log_paths={}, config={},
               before={"ccd": {"exposure_time_s": 0.99999}}, after={"ccd": {}},
               new_files=["results/b.csv", "results/a.csv"])
    assert r2["state"]["before"]["ccd"]["exposure_time_s"] == 0.99999
    assert r2["new_files"] == ["results/a.csv", "results/b.csv"], "kept sorted"

    row = csv_row({**r, "earned": 2, "verdict": "pass", "n_pass": 3, "n_total": 3})
    assert len(row) == len(CSV_COLUMNS), f"{len(row)} != {len(CSV_COLUMNS)}"

    empty = build(item=item, sid="s", wall_s=1.0, llm_calls=[], turn={}, answer="",
                  message="", artifacts=[], setup={}, restore_failed=[],
                  log_paths={}, config={})
    assert empty["timing"]["ttft_s"] is None and empty["llm"]["n_calls"] == 0
    assert empty["llm"]["tok_per_s"] is None and empty["llm"]["ctx_utilization"] is None

    api = [{"step": 1, "wall_s": 4.0, "prompt_tokens": 100, "output_tokens": 100,
            "done_reason": "end_turn", "thinking": "217ch", "text_chars": 9, "n_tool_calls": 0}]
    a = build(item=item, sid="s", wall_s=5.0, llm_calls=api, turn={}, answer="a", message="m",
              artifacts=[], setup={}, restore_failed=[], log_paths={}, config={})
    assert a["timing"]["ttft_s"] is None, "an unmeasurable TTFT must not be written as 0.00"
    assert a["llm"]["tok_per_s"] == 25.0 and a["llm"]["tok_per_s_basis"] == "wall", a["llm"]

    print(f"PASS: timing {len(r['timing'])} fields · llm {len(r['llm'])} fields · "
          f"tools {len(r['tools'])} fields · csv {len(CSV_COLUMNS)} columns · "
          f"empty-turn safe")
