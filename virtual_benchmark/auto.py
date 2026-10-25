
"""Automatic virtual-sample benchmark: run sessions per memory scenario and record each session.

    python -m virtual_benchmark.auto                                   three scenarios × 1 episode
    python -m virtual_benchmark.auto --scenarios off,iso,transfer --episodes 6
    python -m virtual_benchmark.auto --episodes 3 --seeds 0,7,42          sample seed per episode (count = episodes)
    python -m virtual_benchmark.auto --model claude-sonnet-4-6          model (default: config.llm)

Scenarios:
    off        no memory               all config.memory switches off
    iso        memory, no transfer     all switches on; each session starts from an empty memory folder
    transfer   memory with transfer    all switches on; one memory folder accumulates across the scenario
                                       (records and skills of earlier sessions carry over)

One session:
    1 attach a new sample (seed of that episode, no damage history) and new virtual hardware
    2 one agent turn with one prompt (session name vbench_<start time>)
    3 compare the final conditions (last acquisition) with the reference conditions (config.json); no verdict
    4 record to runs/<stamp>_<model>/sessions/<scenario>/<session>.json

Prompts are listed in prompts.json; each prompt is one session, and --episodes N runs the whole
list N times. Episode ep uses the sample with seed seeds[ep] (default RAMAN_VBENCH_SEED + ep), so
scenarios are paired by seed.

Timing:
    total_s          wall clock of the whole session (including reflection)
    llm_s            time waiting for LLM responses (successful attempts) · llm_retry_s failed attempts + retry waits
    tool_pure_s      tool execution time minus the memory share (instrument actions and exposure included)
    memory.*         begin_s (skill/summary injection) · record_s (measurement hook: crop, LPIPS, DB) ·
                     tools_s (sample-context, query and annotation tools) · reflect_s (reflection after the run) · total_s
    other_s          the rest (prompt assembly, log writing, history handling)
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROMPTS_JSON = Path(__file__).parent / "prompts.json"
RUNS_DIR = Path(__file__).parent / "runs"
ARCH = "ReAct"

SCENARIOS = {
    "off": "memory off",
    "iso": "memory on, no transfer across sessions",
    "transfer": "memory on, transfer across sessions",
}

MEMORY_TOOLS = ("set_sample_context", "query_measurements", "annotate_measurement")

_T: dict[str, float] = {}


def _timed(mod, name: str, key: str) -> None:
    """Wrap mod.name to add its elapsed time to _T[key]."""
    fn = getattr(mod, name)

    @functools.wraps(fn)
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _T[key] = _T.get(key, 0.0) + time.perf_counter() - t0
    setattr(mod, name, wrapper)


def _install_timers() -> None:
    """Install timers on every memory hook."""
    from backend.service import memory
    from backend.service.memory import recorder
    _timed(memory, "begin_run", "begin_s")
    _timed(memory, "end_run", "reflect_s")
    for name in ("before_fire", "after_restore", "_write"):
        _timed(recorder, name, "record_s")


def set_scenario(scenario: str, mem_dir: Path) -> None:
    """Set the memory switches and memory folder for a scenario."""
    from backend.common.config import memory as mem_cfg
    from backend.repo.store import memory_db
    on = scenario != "off"
    for k in mem_cfg.COMPONENTS:
        setattr(mem_cfg, k, on)
    memory_db.ROOT = mem_dir


def fresh_rig(seed: int | None = None):
    """Create a fresh sample (no damage history) and new virtual hardware."""
    from backend.common.config import virtual_hw as vhw_cfg
    from virtual_benchmark import rig as rig_mod
    if seed is not None:
        vhw_cfg.VBENCH_SEED = int(seed)
    rig_mod._SAMPLE = None
    return rig_mod.VirtualRig().install()


def memory_state() -> dict:
    """Return what is stored in the current memory folder."""
    from backend.repo.store import memory_db, skill_store
    try:
        n = len(memory_db.measurements(limit=10 ** 6)) if memory_db.db_path().exists() else 0
        return {"n_records": n, "skill_version": skill_store.latest_version()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def gt() -> dict:
    """Return the reference (ground-truth) power and exposure just below the damage threshold."""
    from backend.service.tools.hw_tools.hw_ops import pct_to_mw
    from virtual_benchmark.config import BEST_EXPOSURE_S, BEST_POWER_PCT
    return {"power_pct": BEST_POWER_PCT, "power_mw": pct_to_mw(BEST_POWER_PCT), "exposure_s": BEST_EXPOSURE_S}


def gt_gap(fin: dict | None) -> dict | None:
    """Return how far the final conditions are from the ground truth."""
    if not fin:
        return None
    from virtual_benchmark.sample import dose_of
    g = gt()
    tot = fin["exposure_s"] * fin["n_acc"]
    r = lambda v: round(v, 4)  # noqa: E731
    return {
        "gt": g,
        "power_mw_diff": r(fin["power_mw"] - g["power_mw"]), "power_ratio": r(fin["power_mw"] / g["power_mw"]),
        "exposure_s_diff": r(fin["exposure_s"] - g["exposure_s"]), "exposure_ratio": r(fin["exposure_s"] / g["exposure_s"]),
        "total_exposure_s_diff": r(tot - g["exposure_s"]), "total_exposure_ratio": r(tot / g["exposure_s"]),
        "dose_ratio": r(dose_of(fin["power_pct"], fin["exposure_s"]) * fin["n_acc"]
                        / dose_of(g["power_pct"], g["exposure_s"])),
    }


def timing(total_s: float, turn: dict) -> dict:
    """Return the time breakdown (s) of one session."""
    llm = turn.get("llm_calls") or []
    tools = turn.get("tool_calls") or []
    tool_s = sum(float(c.get("ms") or 0.0) for c in tools) / 1000.0
    mem_tools_s = sum(float(c.get("ms") or 0.0) for c in tools if c.get("name") in MEMORY_TOOLS) / 1000.0
    mem = {"begin_s": _T.get("begin_s", 0.0), "record_s": _T.get("record_s", 0.0),
           "tools_s": mem_tools_s, "reflect_s": _T.get("reflect_s", 0.0)}
    mem["total_s"] = sum(mem.values())
    llm_s = sum(float(c.get("call_took_s") or 0.0) for c in llm)
    llm_retry_s = sum(float(c.get("retry_lost_s") or 0.0) + float(c.get("retry_wait_s") or 0.0) for c in llm)
    tool_pure_s = max(0.0, tool_s - mem["record_s"] - mem_tools_s)
    other = total_s - llm_s - llm_retry_s - tool_s - mem["begin_s"] - mem["reflect_s"]
    r = lambda v: round(v, 3)  # noqa: E731
    return {"total_s": r(total_s), "turn_s": turn.get("turn_took_s"), "llm_s": r(llm_s),
            "llm_retry_s": r(llm_retry_s), "tool_pure_s": r(tool_pure_s),
            "memory": {k: r(v) for k, v in mem.items()}, "other_s": r(max(0.0, other)),
            "n_llm_calls": len(llm), "n_tool_calls": len(tools),
            "input_tokens": sum(int(c.get("input_tokens") or 0) for c in llm),
            "output_tokens": sum(int(c.get("output_tokens") or 0) for c in llm)}


def _dose(fin: dict | None) -> dict | None:
    if not fin:
        return None
    return {k: fin[k] for k in ("power_mw", "power_pct", "exposure_s", "n_acc", "total_exposure_s")}


def _heights(peaks: dict | None) -> dict:
    return {k: v.get("net_height") for k, v in (peaks or {}).items()}


def run_session(scenario: str, prompt: str, agent_fn, seed: int | None = None) -> dict:
    """Run one session: new rig, one turn, comparison to ground truth; never raises."""
    from benchmark.test_pipeline.context import read_turn
    from backend.repo.store import run_store
    from virtual_benchmark import session_log

    sid = run_store.new_session_id()
    _T.clear()
    rec = {"session": sid, "scenario": scenario, "scenario_name": SCENARIOS[scenario],
           "started_at": datetime.now().isoformat(timespec="seconds"), "prompt": prompt,
           "memory_before": memory_state()}
    fresh_rig(seed)
    from backend.common.config import virtual_hw as vhw_cfg
    rec["seed"], rec["scene"] = vhw_cfg.VBENCH_SEED, vhw_cfg.scene_label()
    t0 = time.perf_counter()
    try:
        out = agent_fn(prompt, sid)
        rec["answer"] = str(out.get("final_report") or "")
        rec["error"] = rec["answer"][:300] if rec["answer"].startswith("[Error]") else None
    except Exception as e:  # noqa: BLE001
        rec["answer"], rec["error"] = "", f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    total_s = time.perf_counter() - t0

    from backend.repo.store.paths import RUNS_ROOT
    f = RUNS_ROOT / (run_store.current().get("label") or sid) / session_log.FILE
    shots = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    fin = shots.get("final")
    rec.update({
        "n_shots": shots.get("n_shots", 0),
        "final_ps_peaks": _heights(fin and fin["ps_peaks"]),
        "final_other_peaks": _heights(fin and fin["other_peaks"]),
        "final_dose": _dose(fin),
        "gt_gap": gt_gap(fin),
        "timing": timing(total_s, read_turn(ARCH, sid)),
        "memory_after": memory_state(),
        "final_spot": fin and {"x_mm": fin["x_mm"], "y_mm": fin["y_mm"]},
        "final_peaks_detail": fin and {"ps": fin["ps_peaks"], "other": fin["other_peaks"]},
        "shots": shots.get("shots", []),
    })
    return rec


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


_ROW_PEAKS = (("ps", "1001.4", "ps_1001.4"), ("other", "812 PMMA", "pmma_812"), ("other", "520.7 Si", "si_520.7"))


def session_row(r: dict) -> dict:
    """Return one row of per-session metrics."""
    d, g, t = r.get("final_dose") or {}, r.get("gt_gap") or {}, r["timing"]
    mb, ma = r.get("memory_before") or {}, r.get("memory_after") or {}
    peaks = {"ps": r.get("final_ps_peaks") or {}, "other": r.get("final_other_peaks") or {}}
    return {
        "scenario": r["scenario"], "episode": r.get("episode"), "seed": r.get("seed"), "session": r["session"],
        "error": r["error"],
        "n_shots": r["n_shots"],
        **{k: d.get(k) for k in ("power_mw", "power_pct", "exposure_s", "n_acc", "total_exposure_s")},
        **{k: g.get(k) for k in ("power_ratio", "exposure_ratio", "total_exposure_ratio", "dose_ratio")},
        **{col: peaks[src].get(key) for src, key, col in _ROW_PEAKS},
        "total_s": t["total_s"], "llm_s": t["llm_s"], "llm_retry_s": t["llm_retry_s"],
        "tool_pure_s": t["tool_pure_s"], "memory_s": t["memory"]["total_s"], "reflect_s": t["memory"]["reflect_s"],
        "n_llm_calls": t["n_llm_calls"], "n_tool_calls": t["n_tool_calls"],
        "input_tokens": t["input_tokens"], "output_tokens": t["output_tokens"],
        "records_before": mb.get("n_records"), "records_after": ma.get("n_records"),
        "skill_before": mb.get("skill_version"), "skill_after": ma.get("skill_version"),
    }


def summarize(recs: list[dict]) -> dict:
    """Return per-scenario averages and per-session rows."""
    from virtual_benchmark.config import TARGET_PEAK_CM1
    key = f"{TARGET_PEAK_CM1:g}"
    out = {}
    for sc in SCENARIOS:
        rs = [r for r in recs if r["scenario"] == sc]
        if not rs:
            continue
        t = [r["timing"] for r in rs]
        out[sc] = {
            "name": SCENARIOS[sc], "n_sessions": len(rs),
            "n_error": sum(bool(r["error"]) for r in rs),
            "mean_n_shots": _mean([r["n_shots"] for r in rs]),
            f"mean_ps_{key}": _mean([r["final_ps_peaks"].get(key) for r in rs]),
            "mean_power_mw": _mean([(r["final_dose"] or {}).get("power_mw") for r in rs]),
            "mean_total_exposure_s": _mean([(r["final_dose"] or {}).get("total_exposure_s") for r in rs]),
            "mean_power_ratio": _mean([(r["gt_gap"] or {}).get("power_ratio") for r in rs]),
            "mean_exposure_ratio": _mean([(r["gt_gap"] or {}).get("exposure_ratio") for r in rs]),
            "mean_dose_ratio": _mean([(r["gt_gap"] or {}).get("dose_ratio") for r in rs]),
            "mean_total_s": _mean([x["total_s"] for x in t]),
            "mean_llm_s": _mean([x["llm_s"] for x in t]),
            "mean_tool_pure_s": _mean([x["tool_pure_s"] for x in t]),
            "mean_memory_s": _mean([x["memory"]["total_s"] for x in t]),
            "rows": [session_row(r) for r in rs],
        }
    return out


def _print_row(n: int, total: int, r: dict) -> None:
    from virtual_benchmark.config import TARGET_PEAK_CM1
    d, t, g = r["final_dose"] or {}, r["timing"], r["gt_gap"] or {}
    print(f"  {n:3}/{total}  {r['scenario']:8} {r['session']:24} shots={r['n_shots']:<3} "
          f"PS{TARGET_PEAK_CM1:g}={r['final_ps_peaks'].get(f'{TARGET_PEAK_CM1:g}')}  "
          f"{d.get('power_mw')}mW×{d.get('exposure_s')}s×{d.get('n_acc')} (vs GT P×{g.get('power_ratio')} t×{g.get('exposure_ratio')} dose×{g.get('dose_ratio')})  "
          f"{t['total_s']:.1f}s (llm {t['llm_s']:.1f} · tool {t['tool_pure_s']:.1f} · mem {t['memory']['total_s']:.1f})"
          + (f"  [{r['error']}]" if r["error"] else ""))


def default_seeds(episodes: int) -> list[int]:
    """Return the default sample seed of each episode."""
    from backend.common.config import virtual_hw as vhw_cfg
    return [vhw_cfg.VBENCH_SEED + ep for ep in range(episodes)]


def run(scenarios: list[str], prompts: list[str], episodes: int, agent_fn, out_dir: Path,
        seeds: list[int] | None = None) -> list[dict]:
    """Run episodes × prompts sessions for each scenario in order."""
    seeds = default_seeds(episodes) if seeds is None else list(seeds)
    if len(seeds) != episodes:
        raise ValueError(f"{len(seeds)} seeds ≠ {episodes} episodes")
    _install_timers()
    total = len(scenarios) * episodes * len(prompts)
    recs, n = [], 0
    for sc in scenarios:
        shared = out_dir / "memory" / sc
        for ep in range(episodes):
            for pi, prompt in enumerate(prompts):
                n += 1
                mem_dir = out_dir / "memory" / sc / f"e{ep}_p{pi}" if sc == "iso" else shared
                set_scenario(sc, mem_dir)
                r = run_session(sc, prompt, agent_fn, seeds[ep])
                r["episode"], r["prompt_index"] = ep, pi
                d = out_dir / "sessions" / sc
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{r['session']}.json").write_text(
                    json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                recs.append(r)
                _print_row(n, total, r)
    return recs


def write_summary(out_dir: Path, recs: list[dict], config: dict) -> dict:
    """Write summary.json and summary.csv for all sessions of a run."""
    doc = {"config": config, "scenarios": summarize(recs)}
    (out_dir / "summary.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [row for s in doc["scenarios"].values() for row in s["rows"]]
    if rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return doc


def _config(model: str, episodes: int, n_prompts: int, seeds: list[int]) -> dict:
    return {"arch": ARCH, "model": model, "episodes": episodes, "n_prompts": n_prompts,
            "seeds": list(seeds), "gt": gt()}


def main(argv=None) -> int:
    """Run the virtual-sample benchmark from the command line."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help=f"comma-separated: {','.join(SCENARIOS)}")
    ap.add_argument("--episodes", type=int, default=1, help="number of passes over the prompt list per scenario")
    ap.add_argument("--seeds", default="", help="comma-separated sample seeds, count = episodes. Default: RAMAN_VBENCH_SEED + episode")
    ap.add_argument("--model", default="", help="LLM model. Default: config.llm")
    ap.add_argument("--prompts", default=str(PROMPTS_JSON), help="prompt file (json: {\"prompts\": [...]})")
    args = ap.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    bad = [s for s in scenarios if s not in SCENARIOS]
    if bad:
        print(f"unknown scenario: {bad} (valid: {list(SCENARIOS)})"); return 2
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))["prompts"]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or default_seeds(args.episodes)
    if len(seeds) != args.episodes:
        print(f"{len(seeds)} --seeds ≠ --episodes {args.episodes}"); return 2

    import backend.common.config.llm as llm_cfg
    if args.model:
        llm_cfg.OLLAMA_MODEL = args.model
    model = llm_cfg.OLLAMA_MODEL
    from backend.service.agents.architectures import single_agent_ReAct as agent

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / f"{stamp}_{re.sub(r'[^0-9A-Za-z._-]', '-', model)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(model, args.episodes, len(prompts), seeds)
    print(f"{len(scenarios)} scenarios × {args.episodes} episodes × {len(prompts)} prompts · {model} · "
          f"seeds {seeds}"
          f"  ->  {out_dir.relative_to(_ROOT)}\n")
    recs = run(scenarios, prompts, args.episodes, agent.run_experiment, out_dir, seeds)
    doc = write_summary(out_dir, recs, cfg)

    print("\nBy scenario")
    for sc, s in doc["scenarios"].items():
        print(f"  {sc:8} {s['name']:38} {s['n_sessions']} sessions  shots {s['mean_n_shots']}  "
              f"{s['mean_power_mw']}mW · {s['mean_total_exposure_s']}s  vs GT P×{s['mean_power_ratio']} "
              f"t×{s['mean_exposure_ratio']} dose×{s['mean_dose_ratio']}  "
              f"{s['mean_total_s']}s (llm {s['mean_llm_s']} · tool {s['mean_tool_pure_s']} · mem {s['mean_memory_s']})")
    print(f"\nResults: {(out_dir / 'summary.json').relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
