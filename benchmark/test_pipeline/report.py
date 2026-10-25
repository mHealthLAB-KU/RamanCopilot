
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

from benchmark.test_pipeline.metrics import AXIS_EN, CSV_COLUMNS, csv_row
from backend.agents.logs.reason_log import TRUNCATED_REASONS  # noqa: E402

_MARK = {"pass": "O", "fail": "X", "human": "?", "error": "!"}


def write(out_dir: Path, all_items: list[dict], results: list[dict],
          preflight: dict | None = None) -> None:
    by_id = {i["id"]: i for i in all_items}
    _scores_csv(out_dir / "scores.csv", results)
    _summary_md(out_dir / "summary.md", by_id, results, preflight or {})
    _review_md(out_dir / "review.md", by_id, results)


def _scores_csv(path: Path, results: list[dict]) -> None:
    """One row per item. utf-8-sig so Excel opens it without mangling."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in results:
            w.writerow(csv_row(r))


def _think_n(r) -> int:
    m = re.match(r"(\d+)", str(r["llm"].get("thinking") or ""))
    return int(m.group(1)) if m else 0


def _think_unit(ok: list) -> str:
    for r in ok:
        m = re.match(r"\d+(ch|tok)$", str(r["llm"].get("thinking") or ""))
        if m:
            return m.group(1)
    return ""


def _think_avg(ok: list) -> str:
    ns = [_think_n(r) for r in ok]
    return f"{sum(ns) / len(ns):.0f}{_think_unit(ok)}" if ns else "-"


def _think_max(ok: list) -> str:
    return f"{max((_think_n(r) for r in ok), default=0)}{_think_unit(ok)}"


def _f(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def _summary_md(path: Path, by_id: dict, results: list[dict], preflight: dict) -> None:
    axes = collections.OrderedDict()
    for r in results:
        key = r["axis"]
        ok, tot, pend = axes.setdefault(key, [0, 0, 0])
        axes[key] = [ok + (1 if r["verdict"] == "pass" else 0), tot + 1,
                     pend + (1 if r.get("earned") is None else 0)]

    got = sum(1 for r in results if r["verdict"] == "pass")
    tot = len(results)
    pend_n = sum(1 for r in results if r.get("earned") is None)
    cfg = next((r.get("config") for r in results if r.get("config")), {}) or {}

    svc = [r for r in results if r.get("service_error") is not None]
    n_retried = sum(1 for r in results if r.get("requeued"))

    L = [f"# Benchmark run — {path.parent.name}", "",
         f"**Automatic score {got} / {tot} items correct**"
         + (f"  ·  {pend_n} withheld for human grading" if pend_n else "")
         + (f"  ·  **{len(svc)} never answered (service error)**" if svc else ""), ""]
    if svc:
        L += [f"> **{len(svc)} item(s) hit a transient provider failure that outlived the "
              f"retry budget and were re-run once at the end of the pass, and still failed: "
              f"{', '.join(r['item'] for r in svc)}.** Those items scored 0 without the model "
              f"ever producing an answer, so `{got}/{tot}` understates it by up to "
              f"{len(svc)} item(s); excluding them the score is {got}/{tot - len(svc)}. "
              f"This is a property of the service that hour, not of the model.", ""]
    L += ["## Configuration", "",
         f"- model `{cfg.get('model')}` @ `{cfg.get('host')}`",
         f"- num_ctx {cfg.get('num_ctx')} · llm_timeout {cfg.get('llm_timeout_s')}s "
         f"· thinking {cfg.get('think') or 'n/a'}",
         f"- architecture {cfg.get('arch')} · session isolation "
         f"{cfg.get('chat_isolated')} · grid approval gate {cfg.get('grid_gate')}",
         f"- llm retry {cfg.get('llm_retry') or 'n/a'}"
         + (f" · {n_retried} item(s) re-run after a service failure" if n_retried else ""),
         f"- virtual hardware {cfg.get('virtual_hw')}", ""]

    for grp, warns in (preflight or {}).items():
        L += [f"> **Preflight warning — `{grp}` group**: scores for those items may not "
              f"be trustworthy.", ">"]
        L += [f"> - {w}" for w in warns]
        L.append("")

    from benchmark.test_pipeline.graders import _util as _U
    shown = {k: v for k, v in _U.SPEC_DEVIATIONS.items()
             if any(r["item"] == k for r in results)}
    if shown:
        L += ["## Declared deviations from the specification", "",
              "Every other item is graded exactly as `items.json` states. These are listed",
              "so a relaxed tolerance can never pass unnoticed.", ""]
        for iid, why in sorted(shown.items()):
            L += [f"- **{iid}** — {why}"]
        L.append("")

    L += ["## Score by axis", "",
          "One row per axis. `correct` counts items that passed every checkpoint - there",
          "is no partial credit, so an item with one unmet condition counts as incorrect.", "",
          "| axis | correct | items | ratio | withheld |", "|---|---:|---:|---:|---:|"]
    for k, (g, t, p) in axes.items():
        L.append(f"| {AXIS_EN.get(k, k)} | {g} | {t} | {g / t * 100:.0f}% | {p or ''} |")
    L += ["", f"| **total** | **{got}** | **{tot}** | **{got / tot * 100:.0f}%** | "
              f"**{pend_n}** |", ""]

    ok = [r for r in results if r["llm"]["n_calls"]]
    if ok:
        def avg(f):
            """Average only measured values; None if none."""
            vals = [f(r) for r in ok if f(r) is not None]
            return sum(vals) / len(vals) if vals else None

        def mx(f):
            """Maximum of measured values, keeping None distinct from 0."""
            vals = [f(r) for r in ok if f(r) is not None]
            return max(vals) if vals else None

        bases = {r["llm"].get("tok_per_s_basis") for r in ok} - {None, ""}
        tps_label = ("tokens/s (generation time)" if bases == {"eval"} else
                     "tokens/s (end-to-end, incl. network)" if bases == {"wall"} else
                     "tokens/s (MIXED BASIS - do not compare)" if len(bases) > 1 else
                     "tokens/s")

        L += ["## Latency and tokens (median-free means over "
              f"{len(ok)} items that reached the model)", "",
              "| metric | mean | max |", "|---|---:|---:|",
              f"| time to first token (s) | {_f(avg(lambda r: r['timing']['ttft_s']), 2)} | "
              f"{_f(mx(lambda r: r['timing']['ttft_s']), 2)} |",
              f"| turn wall clock (s) | {_f(avg(lambda r: r['timing']['agent_s']))} | "
              f"{_f(max(r['timing']['agent_s'] for r in ok))} |",
              f"| model generation (s) | {_f(avg(lambda r: r['timing']['llm_eval_s']))} | "
              f"{_f(max(r['timing']['llm_eval_s'] for r in ok))} |",
              f"| tool execution (s) | {_f(avg(lambda r: r['timing']['tool_s']))} | "
              f"{_f(max(r['timing']['tool_s'] for r in ok))} |",
              f"| LLM calls (ReAct steps) | {_f(avg(lambda r: r['llm']['n_calls']))} | "
              f"{max(r['llm']['n_calls'] for r in ok)} |",
              f"| tool calls | {_f(avg(lambda r: r['tools']['n_calls']))} | "
              f"{max(r['tools']['n_calls'] for r in ok)} |",
              f"| prompt tokens (turn total) | {_f(avg(lambda r: r['llm']['prompt_tokens']), 0)} | "
              f"{max(r['llm']['prompt_tokens'] for r in ok)} |",
              f"| output tokens (turn total) | {_f(avg(lambda r: r['llm']['output_tokens']), 0)} | "
              f"{max(r['llm']['output_tokens'] for r in ok)} |",
              f"| thinking (turn total) | {_think_avg(ok)} | {_think_max(ok)} |",
              f"| context utilisation | "
              f"{_f(avg(lambda r: r['llm']['ctx_utilization']), 3)} | "
              f"{_f(max((r['llm']['ctx_utilization'] or 0) for r in ok), 3)} |",
              f"| {tps_label} | {_f(avg(lambda r: r['llm']['tok_per_s']), 1)} | "
              f"{_f(mx(lambda r: r['llm']['tok_per_s']), 1)} |", ""]

        trunc = [r for r in ok if r["llm"]["truncated_calls"]]
        if trunc:
            L += [f"> **{len(trunc)} items had a reply cut off by the output cap** "
                  f"(`done_reason` in {sorted(TRUNCATED_REASONS)}): "
                  + ", ".join(r["item"] for r in trunc) + ".",
                  "> Their answers are incomplete, so the score reflects a truncated reply.", ""]
        heavy = [r for r in ok if (r["llm"]["ctx_utilization"] or 0) > 0.85]
        if heavy:
            L += [f"> **{len(heavy)} items used more than 85% of the context window**: "
                  + ", ".join(r["item"] for r in heavy) + ".",
                  "> Ollama truncates a prompt past num_ctx silently — treat these with care.", ""]

    failed = [r for r in results if r["verdict"] == "fail"]
    if failed:
        L += [f"## Failed items ({len(failed)})", ""]
        for r in failed:
            L.append(f"- **{r['item']}** {by_id[r['item']]['question_en'][:70]}")
            for okc, why in r.get("checkpoints") or []:
                if not okc:
                    L.append(f"    - unmet: {why or '(no description)'}")
        L.append("")

    errs = [r for r in results if r["verdict"] == "error" or r.get("turn_error")]
    if errs:
        L += [f"## Execution failures ({len(errs)}) — these never reached grading", ""]
        for r in errs:
            L.append(f"- **{r['item']}**: {r.get('error') or r.get('turn_error')}")
        L.append("")

    bad_setup = [r for r in results if (r.get("setup") or {}).get("ok") is False]
    if bad_setup:
        L += [f"## Pre-condition setup failed ({len(bad_setup)}) — **scores not trustworthy**", "",
              "For many items the initial state is already the correct answer, so the setup",
              "first puts the instrument into a wrong state. If that failed, the item ran",
              "without any discrimination and a pass means nothing.", ""]
        for r in bad_setup:
            L.append(f"- **{r['item']}**: {(r.get('setup') or {}).get('error')}")
        L.append("")

    bad_restore = [r for r in results if r.get("restore_failed")]
    if bad_restore:
        L += [f"## Restore failed ({len(bad_restore)}) — **later items may be contaminated**", "",
              "If `set_ccd_trigger_mode` restore fails, every later acquisition waits for an",
              "external trigger that never comes. Re-run everything after the first such item.", ""]
        for r in bad_restore:
            L.append(f"- **{r['item']}**: {', '.join(r['restore_failed'])}")
        L.append("")

    L += ["## All items", "",
          "| | item | axis | verdict | checks | wall s | ttft s | gen s | tool s | "
          "calls | tools | in tok | out tok | dose mJ |",
          "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        t, l, tl = r["timing"], r["llm"], r["tools"]
        L.append(f"| {_MARK[r['verdict']]} | {r['item']} | {r['axis']} | {r['verdict']} | "
                 f"{r.get('n_pass', 0)}/{r.get('n_total', 0)} | {_f(t['wall_s'])} | "
                 f"{_f(t['ttft_s'], 2)} | {_f(t['llm_eval_s'])} | {_f(t['tool_s'])} | "
                 f"{l['n_calls']} | {tl['n_calls']} | {l['prompt_tokens']} | "
                 f"{l['output_tokens']} | {r['safety']['dose_mj']} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _review_md(path: Path, by_id: dict, results: list[dict]) -> None:
    """Only what a human must read. No file if there is nothing to review."""
    human = [r for r in results if r.get("earned") is None]
    lenient = [r for r in results if by_id[r["item"]].get("grading_note")]
    if not human and not lenient:
        return

    L = [f"# Human review — {path.parent.name}", ""]

    if human:
        L += [f"## Cannot be graded automatically — {len(human)} items", "",
              "For these, no behaviour, state, artifact or number distinguishes a correct answer",
              "from an incorrect one. Read the answer and mark it correct or incorrect - scoring",
              "is binary, so set `verdict` to pass or fail in scores.csv.", ""]
        for r in human:
            it = by_id[r["item"]]
            L += [f"### {r['item']}", "",
                  f"**Question** — {it['question_en']}", "",
                  f"**Why a human is needed** — {it.get('human_reason', '')}", "",
                  f"**Ground truth** — {it['gt_text']}", "",
                  f"**Automatic evidence** — tools called {r['tools']['order']} · "
                  f"dose {r['safety']['dose_mj']} mJ · "
                  f"{r['llm']['n_calls']} LLM calls · {_f(r['timing']['agent_s'])}s", "",
                  "**Agent answer**", "", "```", (r["io"]["answer"] or "").strip(), "```", "",
                  "Verdict:  [ ] correct    [ ] incorrect", "", "---", ""]

    if lenient:
        L += [f"## Graded automatically but more leniently than the GT — {len(lenient)} items", "",
              "These already have a score. They reliably catch wrong answers but confirm correct",
              "ones only weakly, so the note says what was *not* verified.", "",
              "| item | verdict | not verified |", "|---|---|---|"]
        for r in lenient:
            L.append(f"| {r['item']} | {r['verdict']} | "
                     f"{by_id[r['item']]['grading_note']} |")
        L.append("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import tempfile

    from benchmark.test_pipeline import metrics

    root = Path(__file__).resolve().parents[2]
    items = json.loads((root / "benchmark" / "items.json").read_text(encoding="utf-8"))["items"]
    calls = [{"step": 1, "wall_s": 12.0, "prompt_tokens": 9000, "output_tokens": 300,
              "load_s": 1.5, "prompt_eval_s": 0.8, "eval_s": 9.0, "total_s": 11.5,
              "done_reason": "length", "thinking": "400ch", "text_chars": 200,
              "n_tool_calls": 1}]
    turn = {"total_duration_s": 20.0, "dose_mj": 0.19, "tool_call_order": ["get_ccd_info"],
            "Tool calls": [{"step": 0, "name": "get_ccd_info", "args": {}, "ok": True,
                            "ms": 120.0}]}

    fake = []
    for it in items[:6] + [i for i in items if i["needs_human"]] \
            + [i for i in items if i.get("grading_note")][:2]:
        v = "human" if it["needs_human"] else ("pass" if it["seq"] % 2 else "fail")
        rec = metrics.build(item=it, sid="s", wall_s=21.0, llm_calls=calls, turn=turn,
                            answer="Test answer.", message="Q", artifacts=[],
                            setup={"ok": True}, restore_failed=[], log_paths={},
                            config={"model": "gemma4:31b", "num_ctx": 100000, "think": True})
        rec.update(verdict=v, n_pass=1, n_total=2,
                   earned=None if v == "human" else (it["points"] if v == "pass" else 0),
                   checkpoints=[[True, "met"], [v == "pass", "unmet condition"]])
        fake.append(rec)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write(d, items, fake, preflight={"focus": ["no calibrator"]})
        rows = list(csv.reader((d / "scores.csv").read_text(encoding="utf-8-sig").splitlines()))
        assert rows[0] == CSV_COLUMNS, "csv header drifted from metrics.CSV_COLUMNS"
        assert len(rows) == len(fake) + 1
        assert all(len(r) == len(CSV_COLUMNS) for r in rows), "ragged csv row"

        s = (d / "summary.md").read_text(encoding="utf-8")
        for token in ("Automatic score", "Score by axis", "time to first token",
                      "cut off by the output cap", "Failed items", "All items",
                      "Preflight warning", "no calibrator",
                      "items correct", "| axis | correct | items |",
                      "Declared deviations from the specification"):
            assert token in s, f"summary.md missing: {token}"
        assert "\ub2e8\uc77c\ub3d9\uc791" not in s and "\ud1b5\uacfc" not in s, "summary.md must be English only"
        assert "points" not in s and "earned" not in s, "summary.md must be binary, not points"

        rv = (d / "review.md").read_text(encoding="utf-8")
        assert "Cannot be graded automatically" in rv and "X14" in rv
        assert "more leniently" in rv
    print(f"PASS: scores.csv {len(CSV_COLUMNS)} cols · summary.md (config/axis/latency/"
          f"truncation/failures/preflight) · review.md (human + lenient) · English only")
