
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

INPUT_DIR = _ROOT / "benchmark" / "input"

_COORD_RE = re.compile(r"_x(-?\d+(?:\.\d+)?)_y(-?\d+(?:\.\d+)?)", re.I)

_META_KEYS = {"laser_power_pct": float, "exposure_time": float,
              "laser_nm": float, "mode": str}


def read_spectrum_csv(path: Path) -> tuple[list[float], list[float], dict]:
    """Read a benchmark input CSV as (x, y, conditions)."""
    meta: dict = {}
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(",")
                cast = _META_KEYS.get(k.strip())
                if cast:
                    try:
                        meta[k.strip()] = cast(v.strip())
                    except ValueError:
                        pass
                continue
            rows.append(next(csv.reader([line])))
    if not rows:
        raise ValueError(f"{path.name}: no data rows")

    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    try:
        iy = header.index("intensity")
    except ValueError:
        raise ValueError(f"{path.name}: no intensity column (header {header})")

    grouping = [h for h in header if h in ("spectrum_id", "frame_index")]
    if grouping:
        raise ValueError(f"{path.name}: file contains several spectra ({grouping[0]} column). "
                         f"This function reads single spectra only - attach it for the agent to interpret.")

    ix = next((header.index(h) for h in ("raman_shift_cm-1", "wavelength_nm", "pixel_index")
               if h in header), None)

    meta["_axis"] = header[ix] if ix is not None else None
    x = [float(r[ix]) for r in body if len(r) > max(ix, iy)] if ix is not None else []
    y = [float(r[iy]) for r in body if len(r) > iy]
    return x, y, meta


def _result_from_csv(path: Path) -> dict:
    """Convert one CSV into a result dict in acquire_spectrum format."""
    x, y, meta = read_spectrum_csv(path)
    is_shift = meta.get("_axis") == "raman_shift_cm-1" and bool(x)
    result = {
        "ok": True,
        "mode": meta.get("mode", "single"),
        "length": len(y),
        "data": y,
        "max_intensity": max(y) if y else 0.0,
        "sum_intensity": sum(y),
        "calibrated": is_shift,
        "exposure_time": meta.get("exposure_time"),
        "laser_power_pct": meta.get("laser_power_pct"),
    }
    if is_shift:
        result["raman_shift_cm-1"] = x
    elif x:
        result[meta.get("_axis") or "wavelength_nm"] = x
    return result


def attach(item: dict) -> str:
    """Upload attachments and return the message suffix referencing them."""
    from backend.service.store.upload_store import save_upload

    lines = []
    for name in item["inputs"]:
        src = INPUT_DIR / name
        res = save_upload(name, src.read_bytes())
        lines.append(f"- {name} (file_id: {res['file_id']})")
    return "\n\n[Attached files]\n" + "\n".join(lines) if lines else ""


def measurement(item: dict) -> str:
    """Inject CSVs as measurements made in this session."""
    from backend.service.store.spectrum_store import save_spectrum

    for name in item["inputs"]:
        path = INPUT_DIR / name
        result = _result_from_csv(path)
        m = _COORD_RE.search(name)
        if m:
            meta = {"x": float(m.group(1)), "y": float(m.group(2))}
        else:
            meta = {"tag": path.stem}
        saved = save_spectrum(result, meta)
        if not saved.get("ok"):
            raise RuntimeError(f"{item['id']}: injecting {name} failed - {saved.get('error')}")
    return ""


def history(item: dict) -> str:
    """Place a measurement-history JSON in the session folder and reference it in the message."""
    from backend.service.store import run_store

    lines = []
    for name in item["inputs"]:
        dst = run_store.session_dir() / name
        shutil.copyfile(INPUT_DIR / name, dst)
        rel = f"runs/{run_store.current()['label']}/{name}"
        run_store.record(run_store.KIND_MEASUREMENT, rel,
                         title=f"Measurement history {name}")
        lines.append(f"- {name} (file_id: {rel})")
    return "\n\n[Attached files]\n" + "\n".join(lines) if lines else ""


_ROUTES = {"attach": attach, "measurement": measurement, "history": history}


def apply(item: dict) -> str:
    """Apply the injection for the item's input_kind and return the message suffix."""
    fn = _ROUTES.get(item["input_kind"])
    return fn(item) if fn else ""


_NOT_SINGLE_SPECTRUM = {
    "D15_broken.csv",
    "D18_photobleach.csv",
    "reference_library.csv",
    "reference_library_8.csv",
}


if __name__ == "__main__":
    items = json.loads((_ROOT / "benchmark" / "items.json").read_text(encoding="utf-8"))["items"]

    n_csv = n_meta = 0
    for it in items:
        for name in it["inputs"]:
            p = INPUT_DIR / name
            assert p.is_file(), f"{it['id']}: missing input {name}"
            if p.suffix != ".csv":
                continue
            if name in _NOT_SINGLE_SPECTRUM:
                continue
            x, y, meta = read_spectrum_csv(p)
            assert y, f"{name}: empty intensity"
            assert not x or len(x) == len(y), f"{name}: x({len(x)}) != y({len(y)})"
            n_csv += 1
            n_meta += any(not k.startswith('_') for k in meta)

    for iid in ("P01", "D02", "D08", "D17"):
        it = next(i for i in items if i["id"] == iid)
        assert it["input_kind"] == "measurement", iid
        for name in it["inputs"]:
            r = _result_from_csv(INPUT_DIR / name)
            assert r["ok"] and r["length"] > 0, f"{name}: empty result"
            assert r["calibrated"], f"{name}: raman_shift axis not detected"
    d02 = _result_from_csv(INPUT_DIR / "D02_saturated.csv")
    assert d02["max_intensity"] == 65535, d02["max_intensity"]
    assert d02["laser_power_pct"] == 100 and d02["exposure_time"] == 5.0, "condition comment row not read"
    d17 = next(i for i in items if i["id"] == "D17")
    coords = [_COORD_RE.search(n) for n in d17["inputs"]]
    assert all(coords) and len(coords) == 7, "D17 coordinate parsing failed"
    p05 = _result_from_csv(INPUT_DIR / "P05.csv")
    assert not p05["calibrated"] and "wavelength_nm" in p05, "P05 not read as a nm axis"

    print(f"passed: {n_csv} input CSVs parsed · {n_meta} condition comment rows · "
          f"4 measurement items assembled · D02 saturation · D17 7 coordinates · P05 nm axis")
