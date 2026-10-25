"""Per-session record of laser shots on the virtual sample."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np

from virtual_benchmark.bands import BANDS

FILE = "vbench_shots.json"
_LOCK = threading.Lock()

HALF_WIDTH = 2.0  # cm-1, half-width of the peak window
_PS = [{"center_cm1": c, "half_width_cm1": HALF_WIDTH} for c, _, _ in BANDS["tgt"]]
_OTHER = ([{"center_cm1": c, "half_width_cm1": HALF_WIDTH, "name": "Si"} for c, _, _ in BANDS["bg"]]
          + [{"center_cm1": c, "half_width_cm1": HALF_WIDTH, "name": "PMMA"} for c, _, _ in BANDS["tgt2"]])


def _peaks(axis, y, bands, named=False) -> dict:
    from backend.service.memory.recorder import band_values
    out = {}
    for b, v in zip(bands, band_values(axis, y, bands)):
        key = f"{b['center_cm1']:g}" + (f" {b['name']}" if named else "")
        out[key] = ({"net_height": v.get("net_height"), "snr": v.get("snr")} if v.get("in_range", True)
                    else {"in_range": False})
    return out


def record(axis_cm1, counts, ctx: dict, x_mm: float, y_mm: float) -> None:
    """Append one laser acquisition to the current session file; never blocks the measurement."""
    try:
        from backend.repo.store import run_store
        from backend.service.tools.hw_tools.hw_ops import pct_to_mw
        axis = np.asarray(axis_cm1, float)
        y = np.asarray(counts, float)
        n_acc = max(int(ctx["num_acc"]), 1)
        cond = {"power_pct": round(float(ctx["power_pct"]), 6), "power_mw": pct_to_mw(float(ctx["power_pct"])),
                "exposure_s": float(ctx["exposure_s"]), "n_acc": n_acc,
                "total_exposure_s": round(float(ctx["exposure_s"]) * n_acc, 6),
                "x_mm": round(float(x_mm), 5), "y_mm": round(float(y_mm), 5)}
        final = {**cond, "ps_peaks": _peaks(axis, y, _PS), "other_peaks": _peaks(axis, y, _OTHER, named=True)}
        path = run_store.session_dir() / FILE
        with _LOCK:
            doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
                {"session": run_store.current().get("label") or path.parent.name, "n_shots": 0, "shots": []}
            doc["n_shots"] += 1
            doc["final"] = final
            doc["shots"].append({**cond, "ps_1001": final["ps_peaks"].get("1001.4")})
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] virtual_benchmark.session_log: {type(e).__name__}: {e}", file=sys.stderr)


def table(runs_dir: Path) -> list[dict]:
    """Return the records of session folders, one row each."""
    rows = []
    for f in sorted(Path(runs_dir).glob(f"*/{FILE}")):
        d = json.loads(f.read_text(encoding="utf-8"))
        fin = d.get("final") or {}
        rows.append({"session": d["session"], "n_shots": d["n_shots"],
                     **{k: fin.get(k) for k in ("power_mw", "exposure_s", "n_acc", "total_exposure_s")},
                     "ps_peaks": {k: v.get("net_height") for k, v in (fin.get("ps_peaks") or {}).items()},
                     "other_peaks": {k: v.get("net_height") for k, v in (fin.get("other_peaks") or {}).items()}})
    return rows
