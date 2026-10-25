
"""Benchmark runner: for each item, setup -> run -> grade -> restore.

    python -m benchmark.test_pipeline.run --check          self-check of the harness (no hardware needed)
    python -m benchmark.test_pipeline.run --dry-run        flow only, without hardware (scores meaningless)
    python -m benchmark.test_pipeline.run                  all 140 items
    python -m benchmark.test_pipeline.run --group auto     122 items that need no preparation
    python -m benchmark.test_pipeline.run --group focus    16 items that need focus and calibration
    python -m benchmark.test_pipeline.run --group manual   2 items that need a person to unplug a cable
    python -m benchmark.test_pipeline.run --items S01,S04  selected items only
    python -m benchmark.test_pipeline.run --models gemma4:31b,qwen3:32b
                                                           several models in turn

Steps per item:
    1 begin_session      the harness opens the session first, so injected files and agent
                         artifacts share one folder
    2 inject             attachments / measurement results / history
    3 setup              call TOOL_DISPATCH directly to create the faulty state
    4 snapshot(before)   + list of files under DATA (for path-escape checks)
    5 prompt mode + run_experiment
    6 snapshot(after)    + DetailLog turn entry
    7 grading            full marks only if every checkpoint passes (no partial credit)
    8 restore_baseline   always

Single-threaded: run_store session labels are thread-local and there is one instrument.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.test_pipeline import context, inject, lab, metrics, report  # noqa: E402
from benchmark.test_pipeline.graders import checkpoints  # noqa: E402

ITEMS_JSON = _ROOT / "benchmark" / "items.json"
RUNS_DIR = _ROOT / "benchmark" / "runs"

ARCH = "ReAct"


def _num(x, nd: int = 2, width: int = 0) -> str:
    """Format a number for progress output ('-' for None)."""
    s = f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"
    return s.rjust(width) if width else s


def model_slug(model: str) -> str:
    """Make a model tag safe for paths and session ids ('gemma4:31b' -> 'gemma4-31b')."""
    return re.sub(r"[^0-9A-Za-z._-]", "-", model).strip("-") or "model"


def switch_model(model: str) -> list[str]:
    """Switch the LLM used by this process; return the names of updated modules."""
    import sys as _sys

    import backend.llm_config as cfg
    from backend.agents.runtime import runtime
    from backend.agents.logs import reason_log

    touched = []
    for name, mod in list(_sys.modules.items()):
        if not name.startswith("backend") or mod is None:
            continue
        if getattr(mod, "OLLAMA_MODEL", None) is not None:
            mod.OLLAMA_MODEL = model
            touched.append(name)

    runtime._llm_cache.clear()
    reason_log._think_state = None

    assert runtime.OLLAMA_MODEL == model, "the runtime copy was not updated"
    assert reason_log.OLLAMA_MODEL == model, "the reason_log copy was not updated"
    assert cfg.OLLAMA_MODEL == model, "llm_config was not updated (_config reads it)"
    return sorted(touched)


def load_items() -> list[dict]:
    return json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]


def schedule(items: list[dict]) -> list[dict]:
    """Order items so that no_adjacent pairs are not consecutive."""
    todo, out = list(items), []
    while todo:
        prev = out[-1]["id"] if out else None
        pick = next((i for i, it in enumerate(todo)
                     if prev not in (it["no_adjacent"] or [])
                     and it["id"] not in ((out[-1]["no_adjacent"] or []) if out else [])), 0)
        out.append(todo.pop(pick))
    return out


def _config() -> dict:
    """Return the configuration needed to reproduce a run."""
    import backend.llm_config as cfg
    from backend.agents.logs import reason_log
    think_label, think_why = reason_log.thinking_state()
    return {"arch": ARCH, "model": cfg.OLLAMA_MODEL, "host": cfg.OLLAMA_HOST,
            "num_ctx": cfg.NUM_CTX, "llm_timeout_s": cfg.LLM_TIMEOUT_S,
            "chat_isolated": True, "grid_gate": False,
            "llm_retry": f"{reason_log._RETRY_MAX}x, base {reason_log._RETRY_BASE_S:g}s, "
                         f"cap {reason_log._RETRY_CAP_S:g}s, 5xx/429 only",
            "think": f"{think_label} — {think_why}", "virtual_hw": cfg.VIRTUAL_HW}


def _copy_logs(sid: str, out_dir: Path) -> dict:
    """Copy this item's two logs into the run folder; return relative paths."""
    safe = re.sub(r"[^0-9A-Za-z_-]", "-", sid)[:64]
    out = {}
    for kind, src_dir in (("detail", _ROOT / "DetailLog"),
                          ("reasoning", _ROOT / "DetailLog" / "reasoning")):
        hits = sorted(src_dir.glob(f"{ARCH}_*_{safe}.*"), key=lambda p: p.stat().st_mtime)
        hits = [p for p in hits if p.is_file()]
        if not hits:
            continue
        dst_dir = out_dir / "logs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / hits[-1].name
        shutil.copyfile(hits[-1], dst)
        out[kind] = f"logs/{dst.name}"
    return out


def _relative(paths) -> list:
    """Return a path relative to DATA."""
    from backend.service.store import DATA_ROOT
    out = []
    for p in sorted(paths):
        try:
            out.append(str(p.relative_to(DATA_ROOT)).replace(chr(92), "/"))
        except ValueError:
            out.append(str(p))
    return out


def run_item(item: dict, stamp: str, out_dir: Path, slug: str = "", attempt: int = 1) -> dict:
    """Run one benchmark item; never raises."""
    from backend.agents.architectures import single_agent_ReAct as agent
    from backend.agents.prompts import build_system_prompt
    from backend.agents.logs import reason_log
    from backend.service.store import run_store

    iid = item["id"] + (f"-r{attempt}" if attempt > 1 else "")
    sid = f"bench_{iid}_{slug}_{ARCH}_{stamp}" if slug else f"bench_{iid}_{ARCH}_{stamp}"
    iid = item["id"]
    setup_res: dict = {}
    restore_failed: list = []
    t0 = time.time()

    try:
        run_store.begin_session(sid, ARCH, isolated=True)

        tail = inject.apply(item)

        setup_res = lab.apply_setup(item)

        before, files_before = lab.snapshot(), lab.scan_data()

        agent.SYSTEM_PROMPT = build_system_prompt(
            "ReAct", autonomous=(item["prompt_mode"] == "autonomous"))
        message = item["question_en"] + tail

        reason_log.reset_stats()
        out = agent.run_experiment(message, sid) if not lab.DRY_RUN else {"final_report": ""}
        answer = str(out.get("final_report") or "")
        llm_calls = reason_log.take_stats()

        after, files_after = lab.snapshot(), lab.scan_data()
        turn = context.read_turn(ARCH, sid) if not lab.DRY_RUN else {}

        run = context.Run(item=item, sid=sid, before=before, after=after, answer=answer,
                          turn=turn, files_before=files_before, files_after=files_after,
                          setup=setup_res)

        rec = metrics.build(
            item=item, sid=sid, wall_s=time.time() - t0, llm_calls=llm_calls, turn=turn,
            answer=answer, message=message, artifacts=run.artifacts, setup=setup_res,
            restore_failed=[], log_paths=_copy_logs(sid, out_dir), config=_config(),
            before=before, after=after,
            new_files=_relative(files_after - files_before))

        cps = checkpoints(item, run)
        rec["checkpoints"] = [[bool(ok), why] for ok, why in cps]
        rec["n_pass"] = sum(1 for ok, _ in cps if ok)
        rec["n_total"] = len(cps)
        if item["needs_human"]:
            rec["earned"] = None
            rec["verdict"] = "human"
        else:
            passed = bool(cps) and all(ok for ok, _ in cps)
            rec["earned"] = item["points"] if passed else 0
            rec["verdict"] = "pass" if passed else "fail"

        code = reason_log.transient_failure_in(str(turn.get("error") or ""))
        if code is not None and not item["needs_human"]:
            rec.update(earned=0, verdict="error", service_error=code)

    except Exception as e:  # noqa: BLE001
        rec = metrics.build(item=item, sid=sid, wall_s=time.time() - t0, llm_calls=[],
                            turn={}, answer="", message="", artifacts=[], setup=setup_res,
                            restore_failed=[], log_paths={}, config={})
        rec.update(earned=0, verdict="error", n_pass=0, n_total=0, checkpoints=[],
                   error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    finally:
        t_restore = time.time()
        try:
            restore_failed = lab.restore_baseline()
        except Exception as e:  # noqa: BLE001
            restore_failed = [f"restore itself failed: {e}"]
        restore_s = time.time() - t_restore
    rec["restore_failed"] = restore_failed
    t = rec["timing"]
    t["wall_s"] = round(time.time() - t0, 3)
    t["restore_s"] = round(restore_s, 3)
    t["overhead_s"] = round(max(0.0, t["wall_s"] - t["agent_s"] - restore_s), 3)
    return rec


_SUBCHECKS = ("benchmark.test_pipeline.graders._util", "benchmark.test_pipeline.context",
              "benchmark.test_pipeline.inject", "benchmark.test_pipeline.lab",
              "benchmark.test_pipeline.metrics", "benchmark.test_pipeline.report")


def check() -> int:
    """Self-check of the whole harness (runs without hardware)."""
    import os
    import subprocess
    from benchmark.test_pipeline.graders import selfcheck

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    print("Self-check")
    fails = []
    for mod in _SUBCHECKS:
        p = subprocess.run([sys.executable, "-m", mod], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(_ROOT), env=env)
        tail = (p.stdout or p.stderr or "").strip().splitlines()
        print(f"  {'OK ' if p.returncode == 0 else 'FAIL'} {mod.split('.')[-1]:10} "
              f"{tail[-1][:96] if tail else ''}")
        if p.returncode != 0:
            fails.append(mod)

    print("  OK  graders   ", selfcheck())
    items = load_items()
    print(f"  OK  items      {len(items)} items / {sum(i['points'] for i in items)} points · "
          f"setup {len(lab.SETUP)} items · manual {len(lab.MANUAL_PROMPT)} items")

    sch = schedule(items)
    bad = [(a["id"], b["id"]) for a, b in zip(sch, sch[1:])
           if b["id"] in (a["no_adjacent"] or []) or a["id"] in (b["no_adjacent"] or [])]
    print(f"  {'OK ' if not bad else 'FAIL'} schedule   {len(bad)} adjacency violations {bad[:3]}")
    if bad:
        fails.append("schedule")

    if fails:
        print(f"\nFailed: {', '.join(fails)}")
        return 1
    print("\nAll passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=("auto", "focus", "manual", "all"), default="all",
                    help="auto=no preparation · focus=focus/calibration · manual=human intervention")
    ap.add_argument("--items", default="", help="comma-separated item IDs (e.g. S01,S04)")
    ap.add_argument("--axis", default="", help="axis keys only (e.g. S or S,X)")
    ap.add_argument("--dry-run", action="store_true", help="flow only, without hardware; scores are meaningless")
    ap.add_argument("--check", action="store_true", help="run the harness self-check and exit")
    ap.add_argument("--no-shuffle", action="store_true", help="do not reorder to avoid adjacent pairs")
    ap.add_argument("--models", default="",
                    help="comma-separated model tags, run in order "
                         "(e.g. gemma4:31b,qwen3:32b). Default: the configured model only")
    args = ap.parse_args(argv)

    if args.check:
        return check()

    lab.DRY_RUN = args.dry_run
    items = load_items()
    if args.items:
        want = {s.strip().upper() for s in args.items.split(",") if s.strip()}
        items = [i for i in items if i["id"].upper() in want]
    if args.axis:
        keys = {s.strip().upper() for s in args.axis.split(",") if s.strip()}
        items = [i for i in items if i["axis_key"] in keys]
    if args.group != "all":
        items = [i for i in items if i["group"] == args.group]
    if not items:
        print("no items to run"); return 2
    if not args.no_shuffle:
        items = schedule(items)

    try:
        import backend.tools.tools  # noqa: F401
        import backend.llm_config  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[abort] cannot import the backend: {type(e).__name__}: {e}")
        print("       run with the same interpreter that runs the server:")
        print(r"       backend\.venv\Scripts\python -m benchmark.test_pipeline.run")
        return 4

    lab.boot()
    preflight_warnings = {}
    for grp in sorted({i["group"] for i in items}):
        warns = lab.preflight(grp)
        if warns:
            preflight_warnings[grp] = warns
            print(f"[warning] preconditions of group {grp} not met - scores of those items are unreliable:")
            for w in warns:
                print("   ·", w)
            print()

    import backend.llm_config as _cfg
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [_cfg.OLLAMA_MODEL]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_dirs = []
    for mi, model in enumerate(models, 1):
        slug = model_slug(model)
        touched = switch_model(model)
        if len(models) > 1:
            print(f"[model {mi}/{len(models)}] {model}  ({len(touched)} modules updated)")
        out_dir = RUNS_DIR / f"{stamp}_{slug}"
        (out_dir / "items").mkdir(parents=True, exist_ok=True)
        all_dirs.append((model, out_dir))
        print(f"\n{len(items)} items  ·  {model}  ->  {out_dir.relative_to(_ROOT)}\n")

        def _emit(rec: dict, item: dict, n: int, total: int, tag: str = "") -> None:
            (out_dir / "items" / f"{item['id']}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            mark = {"pass": "O", "fail": "X", "human": "?", "error": "!"}[rec["verdict"]]
            nre = rec["llm"].get("n_retries") or 0
            print(f"  {n:3}/{total}  {mark}  {item['id']:4}{tag} "
                  f"{rec.get('n_pass', 0)}/{rec.get('n_total', 0)}  "
                  f"{rec['timing']['wall_s']:6.1f}s  "
                  f"ttft {_num(rec['timing']['ttft_s'], 2, 5)}s  "
                  f"{rec['llm']['n_calls']}call {rec['tools']['n_calls']}tool  "
                  + (f"[retried {nre}x {rec['llm'].get('retry_codes')}] " if nre else "")
                  + f"{item['question_en'][:40]}")

        results, deferred = [], []
        for n, item in enumerate(items, 1):
            if item["group"] == "manual" and not lab.DRY_RUN:
                print(f"\n  [manual preparation] {item['id']}: {lab.MANUAL_PROMPT.get(item['id'], '')}")
                input("  press Enter when ready: ")
            rec = run_item(item, stamp, out_dir, slug)
            if rec.get("service_error") is not None:
                deferred.append(item)
            results.append(rec)
            _emit(rec, item, n, len(items))

        if deferred:
            print(f"\n  -- retrying {len(deferred)} items deferred due to service errors "
                  f"({', '.join(i['id'] for i in deferred)}) ──")
            for n, item in enumerate(deferred, 1):
                rec = run_item(item, stamp, out_dir, slug, attempt=2)
                rec["requeued"] = True
                results = [r for r in results if r["item"] != item["id"]] + [rec]
                _emit(rec, item, n, len(deferred), tag="*")
            results.sort(key=lambda r: [i["id"] for i in items].index(r["item"]))

        report.write(out_dir, load_items(), results, preflight=preflight_warnings)
        n_ok = sum(1 for r in results if r["verdict"] == "pass")
        print(f"\nResults: {out_dir.relative_to(_ROOT)}/summary.md   ({n_ok}/{len(results)} correct)")
        if any(r["verdict"] == "human" for r in results):
            print(f"      {out_dir.relative_to(_ROOT)}/review.md  <- some items need human grading")

    if len(all_dirs) > 1:
        print("\nResults per model")
        for model, d in all_dirs:
            print(f"  {model:24} {d.relative_to(_ROOT)}")

    if not lab.DRY_RUN:
        lab.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
