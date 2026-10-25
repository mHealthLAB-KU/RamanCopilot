"""Saves measured spectra as CSV, JSON and PNG, grouped by date and session."""
from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from backend.repo.store import DATA_ROOT
from backend.repo.store.paths import RESULTS_ROOT, URL_PREFIX, make_id
from backend.common.result import fail, ok
from backend.common.config.hw import LASER_FULL_MW

UNASSIGNED = "_unassigned"


def session_folder() -> str:
    """Return the current session label used as the measurement subfolder."""
    try:
        from backend.repo.store import run_store
        return run_store.current().get("label") or UNASSIGNED
    except Exception:
        return UNASSIGNED


def _safe_tag(meta: dict | None) -> str:
    """Return a filesystem-safe filename tag from metadata."""
    raw = ""
    if meta:
        if meta.get("tag"):
            raw = str(meta["tag"])
        elif meta.get("x") is not None and meta.get("y") is not None:
            raw = f"x{meta['x']}_y{meta['y']}"
        elif meta.get("sample"):
            raw = str(meta["sample"])
    if not raw:
        raw = "spectrum"
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in raw)
    return safe[:60].strip("_") or "spectrum"


def _x_axis(result: dict) -> tuple[list, str]:
    """Return the x axis (Raman shift or pixel) and its label."""
    if result.get("calibrated") and result.get("raman_shift_cm-1"):
        return list(result["raman_shift_cm-1"]), "Raman shift (cm$^{-1}$)"
    intensity = _intensity_of(result)
    n = result.get("length") or len(intensity)
    return list(range(n)), "Pixel"


def _intensity_of(result: dict) -> list:
    """Return the intensity array of a single/accumulate result."""
    return list(result.get("data") or result.get("intensity") or [])


def _series(result: dict) -> list[tuple[str, list, list, str]]:
    """Return the list of (label, x, y, xlabel) series."""
    if result.get("mode") == "kinetic" and result.get("frames"):
        out = []
        for fr in result["frames"]:
            if fr.get("calibrated") and fr.get("raman_shift_cm-1"):
                x, xlabel = list(fr["raman_shift_cm-1"]), "Raman shift (cm$^{-1}$)"
            else:
                y0 = list(fr.get("intensity") or [])
                x, xlabel = list(range(len(y0))), "Pixel"
            out.append((f"frame {fr.get('frame_index', '?')}",
                        x, list(fr.get("intensity") or []), xlabel))
        return out
    x, xlabel = _x_axis(result)
    return [("", x, _intensity_of(result), xlabel)]


def power_mw_of(result: dict):
    """Return the laser power (mW) of a measurement result."""
    mw = result.get("laser_power_mw")
    if mw is None and result.get("laser_power_pct") is not None:
        mw = round(float(result["laser_power_pct"]) * LASER_FULL_MW / 100.0, 6)
    return mw


def make_title(result: dict, meta: dict | None = None) -> str:
    """Build a human-readable title from the measurement conditions."""
    parts = []
    if meta and meta.get("x") is not None and meta.get("y") is not None:
        parts.append(f"({meta['x']}, {meta['y']})")
    elif meta and meta.get("sample"):
        parts.append(str(meta["sample"]))
    pwr = power_mw_of(result)
    if pwr is not None:
        parts.append(f"{pwr:g}mW")
    exp = result.get("exposure_time")
    if exp is not None:
        parts.append(f"{exp}s")
    mode = result.get("mode")
    if mode == "accumulate" and result.get("num_accumulations"):
        parts.append(f"accum×{result['num_accumulations']}")
    elif mode == "kinetic" and result.get("kinetic_count"):
        parts.append(f"kinetic×{result['kinetic_count']}")
    return " · ".join(parts) if parts else "Raman spectrum"


def render_png(result: dict, path: Path, title: str) -> None:
    """Render a spectrum as PNG."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    series = _series(result)
    xlabel = series[0][3] if series else "Pixel"
    multi = len(series) > 1
    for label, x, y, xl in series:
        if not x or not y:
            continue
        ax.plot(x, y, linewidth=0.9, label=label if multi else None)
        xlabel = xl
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Intensity (ADU)", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.margins(x=0.01)
    if multi:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


_CSV_META_KEYS = ("laser_power_mw", "laser_power_pct", "exposure_time", "laser_nm", "mode")


def write_spectrum_csv(path, intensity, raman_shift=None, wavelength_nm=None,
                       background=None, meta: dict | None = None,
                       encoding: str = "utf-8-sig") -> None:
    """Write one spectrum as CSV in the standard format."""
    ints = [float(v) for v in intensity]
    cols = ["pixel_index"]
    extra = []
    if raman_shift is not None:
        cols.append("raman_shift_cm-1")
        extra.append(("raman_shift_cm-1", [float(v) for v in raman_shift], "{:.3f}"))
    if wavelength_nm is not None:
        cols.append("wavelength_nm")
        extra.append(("wavelength_nm", [float(v) for v in wavelength_nm], "{:.4f}"))
    cols.append("intensity")
    bg = [float(v) for v in background] if background is not None else None
    if bg is not None:
        cols.append("background_intensity")

    for name, seq, _fmt in extra:
        if len(seq) != len(ints):
            raise ValueError(f"{name} has {len(seq)} points but intensity has {len(ints)}.")
    if bg is not None and len(bg) != len(ints):
        raise ValueError(f"background has {len(bg)} points but intensity has {len(ints)}.")

    with open(path, "w", newline="", encoding=encoding) as f:
        for k, v in (meta or {}).items():
            if v is not None:
                f.write(f"# {k},{v}\n")
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(ints)):
            row = [i]
            for _name, seq, fmt in extra:
                row.append(fmt.format(seq[i]))
            row.append(ints[i])
            if bg is not None:
                row.append(bg[i])
            w.writerow(row)


def _write_csv(result: dict, path: Path) -> None:
    """Write a measurement result as CSV."""
    meta = {k: result.get(k) for k in _CSV_META_KEYS}
    if result.get("mode") == "kinetic" and result.get("frames"):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            for k, v in meta.items():
                if v is not None:
                    f.write(f"# {k},{v}\n")
            w = csv.writer(f)
            w.writerow(["frame_index", "pixel_index", "intensity"])
            for fr in result["frames"]:
                for px, val in enumerate(fr.get("intensity") or []):
                    w.writerow([fr.get("frame_index", ""), px, val])
        return
    calibrated = bool(result.get("calibrated") and result.get("raman_shift_cm-1"))
    write_spectrum_csv(
        path,
        intensity=_intensity_of(result),
        raman_shift=result["raman_shift_cm-1"] if calibrated else None,
        wavelength_nm=result.get("wavelength_nm") if calibrated else None,
        meta=meta,
    )


def spectrum_event(tool_result: dict | None) -> dict | None:
    """Build the frontend 'spectrum' event from a saved acquire_spectrum result, or None."""
    saved = (tool_result or {}).get("saved")
    if not saved:
        return None
    if not any(saved.get(k) for k in ("image_url", "csv_url", "zip_url")):
        return None
    return {
        "type": "spectrum",
        "image_url": saved.get("image_url"),
        "title": saved.get("title", ""),
        "csv_url": saved.get("csv_url"),
        "json_url": saved.get("json_url"),
        "zip_url": saved.get("zip_url"),
    }


def save_spectrum(result: dict, meta: dict | None = None) -> dict:
    """Save a measurement as png + csv + json in the date/time folder."""
    if not result or not result.get("ok"):
        return fail("No valid measurement result to save.")
    try:
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        tag = _safe_tag(meta)
        base = f"{stamp}_{tag}"
        sess = session_folder()

        out_dir = RESULTS_ROOT / date_dir / sess
        out_dir.mkdir(parents=True, exist_ok=True)

        png_path = out_dir / f"{base}.png"
        csv_path = out_dir / f"{base}.csv"
        json_path = out_dir / f"{base}.json"

        title = make_title(result, meta)
        render_png(result, png_path, title)
        _write_csv(result, csv_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"title": title, "meta": meta or {}, "session": sess,
                       "timestamp": now.isoformat(), "result": result},
                      f, ensure_ascii=False)

        def url(p: Path) -> str:
            return f"{URL_PREFIX}/{date_dir}/{sess}/{p.name}"

        try:
            from backend.repo.store import run_store
            run_store.record(run_store.KIND_MEASUREMENT, f"results/{date_dir}/{sess}/{base}.csv",
                             title=title, tag=tag, image=f"{base}.png")
        except Exception:
            pass

        return ok(title=title,
                  timestamp=now.isoformat(),
                  session=sess,
                  dir=str(out_dir),
                  files={"png": str(png_path), "csv": str(csv_path), "json": str(json_path)},
                  image_url=url(png_path),
                  csv_url=url(csv_path),
                  json_url=url(json_path))
    except Exception as e:
        return fail(f"Failed to save results: {e}")


def _resolve_date(date: str | None) -> str:
    return date or datetime.now().strftime("%Y-%m-%d")


def _url(*parts: str) -> str:
    return f"{URL_PREFIX}/" + "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))


def run_store_isolated() -> bool:
    """Return whether this turn runs in session-isolated mode."""
    try:
        from backend.repo.store import run_store
        return run_store.isolated_label() is not None
    except Exception:
        return False


def _artifact_dir(day: str) -> tuple[Path, str]:
    """Return (directory, session label) for generated artifacts."""
    sess = session_folder()
    d = RESULTS_ROOT / day / sess
    d.mkdir(parents=True, exist_ok=True)
    return d, sess


def list_results(date: str | None = None, scope: str = "session") -> list[dict]:
    """List saved measurements of a date (default today) in time order."""
    d = _resolve_date(date)
    day_dir = RESULTS_ROOT / d
    if not day_dir.exists():
        return []

    strict = run_store_isolated()
    cur = session_folder() if (strict or scope == "session") else ""
    items = []
    for jp in list(day_dir.glob("*/*.json")) + list(day_dir.glob("*.json")):
        if jp.stem.startswith("_"):
            continue
        sess = jp.parent.name if jp.parent != day_dir else ""
        if cur and sess != cur and (strict or sess):
            continue
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        res = payload.get("result", {}) or {}
        if res.get("laser_power_mw") is None and res.get("laser_power_pct") is not None:
            res["laser_power_mw"] = power_mw_of(res)
        items.append({
            "base": jp.stem, "session": sess, "date": d,
            "title": payload.get("title", ""),
            "timestamp": payload.get("timestamp", ""),
            "meta": payload.get("meta", {}) or {},
            "result": res,
            "png": jp.with_suffix(".png"),
            "csv": jp.with_suffix(".csv"),
            "json": jp,
        })
    items.sort(key=lambda it: (it["timestamp"] or "", it["base"]))
    return items


def _select(date: str | None, names: list[str] | None,
            scope: str = "session") -> tuple[str, list[dict]]:
    d = _resolve_date(date)
    items = list_results(d, scope=scope)
    if names:
        want = set(names)
        items = [it for it in items if it["base"] in want]
    return d, items


def combine_spectra(date: str | None = None, names: list[str] | None = None,
                    out_name: str | None = None, max_cols: int = 4,
                    scope: str = "session") -> dict:
    """Render several saved spectra into one grid image."""
    d, items = _select(date, names, scope)
    if not items:
        return fail(f"No measurement results to combine on {d}.")
    n = len(items)
    cols = max(1, min(max_cols, n))
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5),
                             dpi=110, squeeze=False)
    for idx, it in enumerate(items):
        ax = axes[idx // cols][idx % cols]
        xlabel = "Pixel"
        for _lbl, x, y, xl in _series(it["result"]):
            if x and y:
                ax.plot(x, y, linewidth=0.7)
                xlabel = xl
        ax.set_title(it["title"] or it["base"], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(xlabel, fontsize=6)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{d} · {n} measurements combined", fontsize=12)
    fig.tight_layout()

    out_dir, sess = _artifact_dir(d)
    stamp = datetime.now().strftime("%H%M%S")
    base = (out_name or f"combined_{stamp}").lstrip("_")
    png = out_dir / f"_{base}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return ok(count=n,
              file_id=make_id("results", d, sess, png.name),
              saved={"title": f"Combined · {n} measurements", "image_url": _url(d, sess, png.name)})

def aggregate_spectra_csv(date: str | None = None, names: list[str] | None = None,
                          out_name: str | None = None, scope: str = "session") -> dict:
    """Write a CSV summarizing saved measurements, one row per experiment."""
    d, items = _select(date, names, scope)
    if not items:
        return fail(f"No measurement results to summarize on {d}.")
    out_dir, sess = _artifact_dir(d)
    stamp = datetime.now().strftime("%H%M%S")
    base = (out_name or f"summary_{stamp}").lstrip("_")
    csv_path = out_dir / f"_{base}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "title", "x", "y", "power_mw", "exposure_s",
                    "mode", "max_intensity", "sum_intensity", "peak_raman_shift_cm-1"])
        for it in items:
            res, meta = it["result"], it["meta"]
            inten = _intensity_of(res)
            peak = ""
            if res.get("calibrated") and res.get("raman_shift_cm-1") and inten:
                pk = max(range(len(inten)), key=lambda i: inten[i])
                peak = f"{res['raman_shift_cm-1'][pk]:.2f}"
            w.writerow([
                d, (it["timestamp"][11:19] if len(it["timestamp"]) >= 19 else ""),
                it["title"], meta.get("x", ""), meta.get("y", ""),
                power_mw_of(res) if power_mw_of(res) is not None else "", res.get("exposure_time", ""),
                res.get("mode", ""),
                f"{max(inten):.1f}" if inten else "",
                f"{sum(inten):.1f}" if inten else "", peak,
            ])
    return ok(count=len(items),
              file_id=make_id("results", d, sess, csv_path.name),
              saved={"title": f"Summary table · {len(items)} measurements",
                     "csv_url": _url(d, sess, csv_path.name)})

def save_scene(image, extent: list | None = None, meta: dict | None = None) -> dict:
    """Save a microscope image with its stage extent for overlays."""
    import numpy as np
    try:
        arr = np.asarray(image)
        day = datetime.now().strftime("%Y-%m-%d")
        sess = session_folder()
        out_dir = RESULTS_ROOT / day / sess
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        npz_path = out_dir / f"_scene_{stamp}.npz"
        png_path = out_dir / f"_scene_{stamp}.png"
        ext_arr = np.asarray(extent, dtype=float) if extent else np.asarray([], dtype=float)
        np.savez_compressed(npz_path, image=arr, extent=ext_arr)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None,
                  extent=(extent if extent else None))
        ax.set_title("Microscope image")
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        return ok(image_url=f"{URL_PREFIX}/{day}/{sess}/{png_path.name}",
                  session=sess,
                  scene_npz=str(npz_path),
                  extent=extent)
    except Exception as e:
        return fail(f"Failed to save microscope image: {e}")


def save_preview_png(png_bytes: bytes, tag: str = "preview") -> dict:
    """Save already-encoded PNG bytes and return the image_url."""
    try:
        day = datetime.now().strftime("%Y-%m-%d")
        out_dir, sess = _artifact_dir(day)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        safe = "".join(c for c in str(tag) if c.isalnum() or c in "-_") or "preview"
        name = f"_{safe}_{stamp}.png"
        (out_dir / name).write_bytes(png_bytes)
        return ok(image_url=_url(day, sess, name), file_id=make_id("results", day, sess, name))
    except Exception as e:
        return fail(f"Failed to save preview image: {e}")


def latest_scene(date: str | None = None, scope: str = "session") -> str | None:
    """Return the most recent scene npz path of this session, or None."""
    day_dir = RESULTS_ROOT / _resolve_date(date)
    if not day_dir.exists():
        return None
    strict = run_store_isolated()
    if strict or scope == "session":
        sess_dir = day_dir / session_folder()
        scenes = sorted(sess_dir.glob("_scene_*.npz")) if sess_dir.exists() else []
        if scenes:
            return str(scenes[-1])
        if strict:
            return None
        scenes = sorted(day_dir.glob("_scene_*.npz"))
        return str(scenes[-1]) if scenes else None
    scenes = sorted(day_dir.glob("_scene_*.npz")) + sorted(day_dir.glob("*/_scene_*.npz"))
    scenes.sort(key=lambda p: p.name)
    return str(scenes[-1]) if scenes else None


def bundle_results(date: str | None = None, names: list[str] | None = None,
                   include: tuple[str, ...] = ("png", "csv", "json"),
                   scope: str = "session") -> dict:
    """Zip saved measurement files and return a download URL."""
    d, items = _select(date, names, scope)
    if not items:
        return fail(f"No measurement results to bundle on {d}.")
    out_dir, sess = _artifact_dir(d)
    stamp = datetime.now().strftime("%H%M%S")
    zip_name = f"_bundle_{stamp}.zip"
    zip_path = out_dir / zip_name
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for it in items:
            for ext in include:
                p = it[ext]
                if isinstance(p, Path) and p.exists():
                    arc = f"{it['session']}/{p.name}" if it.get("session") else p.name
                    z.write(p, arcname=arc)
                    n_files += 1
    return ok(count=len(items),
              files=n_files,
              saved={"title": f"Bundle (zip) · {len(items)} measurements / {n_files} files",
                     "zip_url": _url(d, sess, zip_name)})
