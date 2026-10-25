"""Sandbox that runs agent-written analysis code on stored measurements."""
from __future__ import annotations

import ast
import builtins as _builtins
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODULE_PATH = "backend.service.analyse.analysis_sandbox"

from backend.common.result import fail, ok  # noqa: E402

if __name__ != "__main__":
    assert __name__ == _MODULE_PATH, (
        f"analysis_sandbox was moved: _MODULE_PATH={_MODULE_PATH!r} but the actual module is "
        f"{__name__!r}. Update _MODULE_PATH and the parents[] depth of _PROJECT_ROOT together.")
    assert (_PROJECT_ROOT / "backend").is_dir(), (
        f"no backend/ under _PROJECT_ROOT={_PROJECT_ROOT} - the parents[] depth is wrong. "
        f"Otherwise the child process dies immediately with \"No module named 'backend'\".")

_ALLOWED_IMPORT_ROOTS = {
    "numpy", "scipy", "math", "statistics", "matplotlib", "pandas",
    "json", "itertools", "functools", "collections", "cmath", "random",
}
_BANNED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open", "eval",
    "exec", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "exit", "quit", "breakpoint", "help",
    "memoryview", "classmethod", "staticmethod", "super", "object", "type",
}
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bytes", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "int", "isinstance",
    "issubclass", "len", "list", "map", "max", "min", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "abs", "hasattr",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "RuntimeError", "StopIteration", "ArithmeticError",
]


_MAX_STDOUT_CHARS = 4000  # characters of stdout returned

_MAX_SAVED_FILES = 20  # files save_result may write per run


def validate_code(code: str) -> None:
    """Reject unsafe code before it runs.

    Args:
        code (str): Python source.

    Raises:
        ValueError: If the code uses a disallowed import, name or construct.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"Disallowed import: '{a.name}'. "
                                     f"Allowed: {sorted(_ALLOWED_IMPORT_ROOTS)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed import: '{node.module}'")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                raise ValueError(f"Disallowed attribute access: '{node.attr}' "
                                 "(dunder attributes forbidden)")
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                raise ValueError(f"Disallowed name use: '{node.id}'")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal not allowed")


def _safe_builtins() -> dict:
    d = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(_builtins, name):
            d[name] = getattr(_builtins, name)
    def _guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"Disallowed import: {name}")
        return _builtins.__import__(name, *args, **kwargs)
    d["__import__"] = _guarded_import
    return d


def _clip_stdout(s: str) -> tuple[str, bool]:
    """Truncate stdout to the limit; return (text, truncated)."""
    if len(s) <= _MAX_STDOUT_CHARS:
        return s, False
    return (
        s[:_MAX_STDOUT_CHARS]
        + f"\n\n...[stdout truncated at {_MAX_STDOUT_CHARS} chars of {len(s)} total]\n"
          "Do NOT print large arrays - the output is cut off and it wastes the context window. "
          "Save arrays with save_result(filename, intensity, raman_shift=...) instead; "
          "it returns the saved path. print() only small summaries (counts, peak positions, statistics).",
        True,
    )


def _sanitize_filename(name) -> str:
    """Reduce a save_result filename to a single safe .csv name."""
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("save_result: filename is empty.")
    tail = raw.replace("\\", "/").rstrip("/").split("/")[-1]
    tail = tail.replace(":", "_")
    safe = "".join(c for c in tail if c.isalnum() or c in "._- ").strip(" .")
    if not safe:
        raise ValueError(f"save_result: filename {raw!r} has no usable characters.")
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    return safe


def _make_save_result(data_dir, saved: list, rel_prefix: str = "", start_index: int = 1):
    """Create the save_result hook injected into the sandbox."""
    from backend.repo.store.spectrum_store import write_spectrum_csv as _write_spectrum_csv

    def save_result(filename, intensity, raman_shift=None, wavelength_nm=None,
                    metadata=None) -> str:
        """Save a computed spectrum to the session folder and return its path relative to DATA."""
        if len(saved) >= _MAX_SAVED_FILES:
            raise RuntimeError(
                f"save_result: too many files in one run (max {_MAX_SAVED_FILES}).")

        stem = _sanitize_filename(filename)[:-4]
        safe = f"{start_index + len(saved):02d}_{stem}.csv"

        def _col(v, label):
            """Validate one optional column argument."""
            if v is None:
                return None
            seq = [float(x) for x in v]
            if not seq:
                raise ValueError(f"save_result: {label} is empty.")
            return seq

        ints = _col(intensity, "intensity")
        if ints is None:
            raise ValueError("save_result: intensity is required.")
        shift = _col(raman_shift, "raman_shift")
        wl = _col(wavelength_nm, "wavelength_nm")
        for label, seq in (("raman_shift", shift), ("wavelength_nm", wl)):
            if seq is not None and len(seq) != len(ints):
                raise ValueError(
                    f"save_result: {label} has {len(seq)} points but intensity has "
                    f"{len(ints)} - they must match.")

        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / safe
        n = 2
        while path.exists():
            path = data_dir / f"{safe[:-4]}-{n}.csv"
            n += 1
        safe = path.name
        _write_spectrum_csv(path, intensity=ints, raman_shift=shift, wavelength_nm=wl)

        if metadata:
            try:
                path.with_suffix(".json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
            except Exception:
                pass

        rel = f"{rel_prefix}{safe}" if rel_prefix else safe
        saved.append({"path": rel, "num_points": len(ints),
                      "has_raman_shift": shift is not None})
        return rel

    return save_result


def _main(payload_path: str) -> None:
    import io
    import contextlib
    import traceback
    from datetime import datetime

    from backend.repo.store.spectrum_store import RESULTS_ROOT, URL_PREFIX

    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    code = payload["code"]

    buf = io.StringIO()
    result: dict = {"ok": False}
    saved_files: list = []
    try:
        validate_code(code)

        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        spectra = []
        for it in payload["spectra"]:
            d = dict(it)
            d["intensity"] = np.asarray(it.get("intensity") or [], dtype=float)
            rs = it.get("raman_shift")
            d["raman_shift"] = np.asarray(rs, dtype=float) if rs else None
            wl = it.get("wavelength_nm")
            if wl:
                d["wavelength_nm"] = np.asarray(wl, dtype=float)
            if "frames" in d:
                fr = d["frames"]
                if it.get("frames_ragged"):
                    d["frames"] = [np.asarray(f, dtype=float) for f in fr]
                elif fr:
                    d["frames"] = np.asarray(fr, dtype=float).reshape(len(fr), -1)
                    if d["intensity"].size == 0:
                        d["intensity"] = d["frames"].mean(axis=0)
                        d["intensity_is_frame_mean"] = True
                else:
                    d["frames"] = np.zeros((0, 0), dtype=float)
            spectra.append(d)

        microscope_image = None
        image_extent = None
        scene_path = payload.get("scene_path")
        if scene_path:
            with np.load(scene_path) as z:
                microscope_image = z["image"]
                ext = z["extent"]
                image_extent = [float(v) for v in ext] if ext.size == 4 else None

        files = []
        for up in payload.get("uploads") or []:
            table = {}
            for cname, vals in (up.get("numeric") or {}).items():
                table[cname] = np.asarray(
                    [np.nan if v is None else v for v in vals], dtype=float)
            for cname, vals in (up.get("text") or {}).items():
                table[cname] = list(vals)
            files.append({
                "file_id": up.get("file_id"),
                "filename": up.get("filename"),
                "sheet": up.get("sheet"),
                "columns": up.get("columns") or [],
                "n_rows": up.get("n_rows", 0),
                "table": table,
            })

        _sdir = payload.get("spectra_dir")
        save_result = _make_save_result(
            Path(_sdir) if _sdir else (_PROJECT_ROOT / "DATA"),
            saved_files,
            rel_prefix=payload.get("spectra_rel_prefix") or "",
            start_index=int(payload.get("spectra_start_index") or 1),
        )

        from backend.service.analyse.spectro_math import ipbsa as _ipbsa_fn, poly_baseline as _poly_bl

        ns = {
            "__builtins__": _safe_builtins(),
            "np": np, "plt": plt, "spectra": spectra, "files": files,
            "microscope_image": microscope_image, "image_extent": image_extent,
            "save_result": save_result,
            "ipbsa": _ipbsa_fn, "poly_baseline": _poly_bl,
        }
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<analysis>", "exec"), ns)  # noqa: S102

        day = datetime.now().strftime("%Y-%m-%d")
        sess = payload.get("session_folder") or "_unassigned"
        out_dir = RESULTS_ROOT / day / sess
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        images = []
        for i, num in enumerate(plt.get_fignums()):
            fig = plt.figure(num)
            name = f"fig{stamp}_{i}.png"
            fig.savefig(out_dir / name, bbox_inches="tight")
            images.append(f"{URL_PREFIX}/{day}/{sess}/{name}")
        plt.close("all")

        out, clipped = _clip_stdout(buf.getvalue())
        result = ok(stdout=out, images=images, saved_files=saved_files)
        if clipped:
            result["stdout_truncated"] = True
    except Exception as e:
        out, clipped = _clip_stdout(buf.getvalue())
        result = fail(f"{type(e).__name__}: {e}",
                      stdout=out,
                      trace=traceback.format_exc()[-1200:],
                      saved_files=saved_files)
        if clipped:
            result["stdout_truncated"] = True
    sys.stdout.write("\n__ANALYSIS_RESULT__" + json.dumps(result))


_MAX_KINETIC_FRAMES = 200  # frames loaded per kinetic measurement


def _kinetic_payload(res: dict) -> dict:
    """Build the sandbox payload for a kinetic result (empty dict otherwise)."""
    if res.get("mode") != "kinetic":
        return {}
    frames = res.get("frames") or []
    if not frames:
        return {"n_frames": 0, "frames": []}

    total = len(frames)
    kept = frames[:_MAX_KINETIC_FRAMES]
    lengths = {len(fr.get("intensity") or []) for fr in kept}
    out: dict = {
        "n_frames": len(kept),
        "n_frames_total": total,
        "frames": [list(fr.get("intensity") or []) for fr in kept],
    }
    if len(lengths) == 1:
        first = kept[0]
        if not res.get("raman_shift_cm-1") and first.get("raman_shift_cm-1"):
            out["raman_shift"] = list(first["raman_shift_cm-1"])
        if first.get("wavelength_nm"):
            out["wavelength_nm"] = list(first["wavelength_nm"])
    else:
        out["frames_ragged"] = True
    if total > len(kept):
        out["frames_truncated"] = (
            f"Only the first {len(kept)} of {total} frames were loaded "
            f"(limit {_MAX_KINETIC_FRAMES} per measurement).")
    return out


def run_analysis(code: str, date: str | None = None, names: list[str] | None = None,
                 title: str | None = None, timeout_sec: int = 60,
                 file_ids: list[str] | None = None) -> dict:
    """Run analysis or plotting code on stored measurements and attached files.

    The code runs in a separate process with restricted imports and builtins.

    Args:
        code (str): Python source.
        date (str, optional): Measurement date. Defaults to today.
        names (list[str], optional): Measurements to load. All of the date if omitted.
        title (str, optional): Title of the saved figure.
        timeout_sec (int): Time limit in seconds.
        file_ids (list[str], optional): Attached files to load into the files variable.

    Returns:
        dict: stdout, image count, saved figure and files, or an error.
    """
    import os
    import subprocess
    import tempfile
    from backend.repo.store.spectrum_store import (list_results, latest_scene,
                                        session_folder as _session_folder)
    from backend.repo.store.upload_store import load_upload

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()]

    if not isinstance(code, str) or not code.strip():
        return fail("code is empty. Provide the Python analysis code to run.")
    date = (date.strip() or None) if isinstance(date, str) else date
    title = (title.strip() or None) if isinstance(title, str) else title
    names = _as_list(names) or None
    file_ids = _as_list(file_ids)

    items = list_results(date)
    if names:
        want = set(names)
        items = [it for it in items if it["base"] in want]

    spectra = []
    for it in items:
        res = it["result"]
        intensity = res.get("data") or res.get("intensity") or []
        entry = {
            "base": it["base"], "title": it["title"],
            "x": it["meta"].get("x"), "y": it["meta"].get("y"),
            "power_mw": res.get("laser_power_mw"), "exposure": res.get("exposure_time"),
            "mode": res.get("mode"),
            "raman_shift": res.get("raman_shift_cm-1"),
            "intensity": list(intensity),
        }
        entry.update(_kinetic_payload(res))
        spectra.append(entry)

    uploads = []
    for fid in file_ids:
        try:
            uploads.append(load_upload(str(fid)))
        except Exception as e:
            return fail(f"Could not load the attached file '{fid}': {type(e).__name__}: {e}",
                        hint="Check the exact file_id with list_uploaded_files.")

    try:
        validate_code(code)
    except ValueError as e:
        return fail(f"Code policy violation: {e}",
                    hint="Analysis only with numpy/scipy/matplotlib. No hardware/file/network access.")

    from backend.repo.store import run_store
    _spectra_dir = run_store.session_dir() / run_store.KIND_SPECTRA
    _rel_prefix = f"{run_store.current()['rel_dir']}/{run_store.KIND_SPECTRA}/"
    payload = {"code": code, "spectra": spectra, "scene_path": latest_scene(date),
               "uploads": uploads,
               "spectra_dir": str(_spectra_dir),
               "spectra_rel_prefix": _rel_prefix,
               "spectra_start_index": run_store.next_index(run_store.KIND_SPECTRA),
               "session_folder": _session_folder()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = f.name

    _cwd = Path(_spectra_dir).parent
    try:
        _cwd.mkdir(parents=True, exist_ok=True)
    except OSError:
        _cwd = _PROJECT_ROOT / "DATA"
        _cwd.mkdir(parents=True, exist_ok=True)

    _env = dict(os.environ)
    _env["PYTHONPATH"] = (str(_PROJECT_ROOT) + os.pathsep + _env.get("PYTHONPATH", "")
                          ).rstrip(os.pathsep)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", _MODULE_PATH, payload_path],
            capture_output=True, text=True, errors="replace", timeout=timeout_sec,
            cwd=str(_cwd), env=_env,
        )
    except subprocess.TimeoutExpired:
        return fail(f"Execution timed out ({timeout_sec}s) - infinite loop or too heavy a computation.")
    finally:
        try:
            Path(payload_path).unlink()
        except OSError:
            pass

    marker = "__ANALYSIS_RESULT__"
    out = proc.stdout or ""
    if marker not in out:
        tail = (proc.stderr or out)[-800:]
        return fail("The analysis process exited without a result.", detail=tail)
    payload = json.loads(out.split(marker, 1)[1])

    if not payload.get("ok"):
        err = fail(payload.get("error", "Analysis failed"),
                   stdout=payload.get("stdout", ""),
                   trace=payload.get("trace", ""))
        if payload.get("saved_files"):
            err["saved_files"] = payload["saved_files"]
            for _sf in payload["saved_files"]:
                run_store.record(run_store.KIND_SPECTRA, _sf.get("path", ""),
                                 num_points=_sf.get("num_points"), partial=True)
        if payload.get("stdout_truncated"):
            err["stdout_truncated"] = True
        return err

    images = payload.get("images", [])
    resp = ok(stdout=payload.get("stdout", ""), image_count=len(images), images=images)
    if payload.get("stdout_truncated"):
        resp["stdout_truncated"] = True
    if payload.get("saved_files"):
        resp["saved_files"] = payload["saved_files"]
        for _sf in payload["saved_files"]:
            run_store.record(run_store.KIND_SPECTRA, _sf.get("path", ""),
                             num_points=_sf.get("num_points"),
                             has_raman_shift=_sf.get("has_raman_shift"))
    for _img in images:
        run_store.record(run_store.KIND_FIGURE, _img, title=title or "Analysis result")
    if images:
        resp["saved"] = {"title": title or "Analysis result", "image_url": images[0]}
    return resp


if __name__ == "__main__":
    _main(sys.argv[1])
