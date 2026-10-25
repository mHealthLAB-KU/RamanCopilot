"""Saves and parses data files attached by the user."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from backend.repo.store.paths import UPLOADS_ROOT, FileIdError, make_id, resolve

_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_SUFFIXES = _TABLE_SUFFIXES | _IMAGE_SUFFIXES
_EXCEL_SUFFIXES = {".xlsx", ".xls"}

_SIDECAR_EXT = ".json"

_MAX_LOAD_ROWS = 200_000  # row limit when loading an attached table
_MAX_REPORT_COLS = 32  # column limit in a table summary


def _resolve_date(date: str | None) -> str:
    return date or datetime.now().strftime("%Y-%m-%d")


def _safe_name(raw: str) -> str:
    """Return a filesystem-safe version of the original file name, keeping the extension."""
    p = Path(raw or "")
    suffix = p.suffix.lower()
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in p.stem)
    stem = stem.strip("_")[:60] or "upload"
    return stem + suffix


def _resolve_path(file_id: str) -> Path:
    """Resolve a file_id to a path via paths.resolve."""
    try:
        return resolve(file_id, kind="table")
    except FileIdError as e:
        msg = str(e)
        if "Not found" in msg or "No file matches" in msg or "File not found" in msg:
            raise FileNotFoundError(msg) from None
        raise


def resolve_upload_path(file_id: str) -> Path:
    """Resolve a file_id to a path (public entry point for other modules)."""
    return _resolve_path(file_id)


def _kind(path: Path) -> str:
    """Return the file kind shown in listings."""
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _EXCEL_SUFFIXES:
        return "excel"
    return "text"


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + _SIDECAR_EXT)


def _write_sidecar(path: Path, file_id: str, original: str) -> None:
    """Write the metadata sidecar after saving."""
    try:
        _sidecar_path(path).write_text(json.dumps({
            "file_id": file_id,
            "original_filename": original,
            "stored_filename": path.name,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "bytes": path.stat().st_size,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[upload_store] sidecar write failed for {path.name}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)


def read_sidecar(path: Path) -> dict:
    """Return the metadata sidecar of a file, or an empty dict."""
    try:
        with open(_sidecar_path(path), encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_upload(filename: str, data: bytes) -> dict:
    """Save uploaded bytes in today's folder and return the file_id."""
    name = _safe_name(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    day = _resolve_date(None)
    out_dir = UPLOADS_ROOT / day
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
    stored = f"{stamp}_{name}"
    path = out_dir / stored
    path.write_bytes(data)

    file_id = make_id("uploads", day, stored)
    _write_sidecar(path, file_id, str(filename or ""))

    return {
        "ok": True,
        "file_id": file_id,
        "filename": name,
        "original_filename": str(filename or ""),
        "kind": _kind(path),
        "bytes": path.stat().st_size,
    }


def list_uploads(date: str | None = None) -> list[dict]:
    """List files uploaded on a date (default today) in time order."""
    day = _resolve_date(date)
    day_dir = UPLOADS_ROOT / day
    if not day_dir.exists():
        return []
    items = []
    for p in sorted(day_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        side = read_sidecar(p)
        item = {
            "file_id": make_id("uploads", day, p.name),
            "filename": p.name,
            "kind": _kind(p),
            "bytes": p.stat().st_size,
        }
        original = side.get("original_filename")
        if original and original != p.name:
            item["original_filename"] = original
        items.append(item)
    return items


def _looks_numeric(values) -> bool:
    """Return whether all values in a row parse as numbers."""
    seen = False
    for v in values:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            continue
        try:
            float(s)
        except ValueError:
            return False
        seen = True
    return seen


def _read_text_table(path: Path):
    """Read csv/tsv/txt/dat into a DataFrame, detecting delimiter and header."""
    import pandas as pd

    attempts = (
        {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
        {"sep": r"\s+", "engine": "python", "encoding": "utf-8-sig"},
        {"sep": ",", "engine": "python", "encoding": "utf-8-sig"},
    )
    best = None
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            probe = pd.read_csv(path, nrows=1, header=None, comment="#", **kwargs)
            header = None if _looks_numeric(probe.iloc[0].tolist()) else 0
            df = pd.read_csv(path, header=header, comment="#", **kwargs)
        except Exception as e:
            last_err = e
            continue
        if df.shape[1] == 0 or df.shape[0] == 0:
            continue
        if header is None:
            df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
        if best is None or df.shape[1] > best.shape[1]:
            best = df
        if best.shape[1] > 1:
            break

    if best is None:
        raise ValueError(f"Could not parse as a table: {last_err}")
    return best


def _read_excel_table(path: Path, sheet: str | None):
    """Read one sheet of an xlsx/xls file into a DataFrame."""
    import pandas as pd

    xls = pd.ExcelFile(path)
    sheets = list(xls.sheet_names)
    target = sheet if (sheet and sheet in sheets) else (sheets[0] if sheets else 0)

    probe = pd.read_excel(xls, sheet_name=target, header=None, nrows=1)
    header = None if (len(probe) and _looks_numeric(probe.iloc[0].tolist())) else 0
    df = pd.read_excel(xls, sheet_name=target, header=header)
    if header is None:
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    return df, sheets, str(target)


def _read_table(path: Path, sheet: str | None):
    if _kind(path) == "excel":
        return _read_excel_table(path, sheet)
    return _read_text_table(path), [], None


def _is_numeric_col(series) -> bool:
    import pandas as pd
    return bool(pd.api.types.is_numeric_dtype(series))


def _cell(v):
    """Return a JSON-safe value for the head preview."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)[:80]
    if f != f:
        return None
    return round(f, 6)


def inspect_upload(file_id: str, sheet: str | None = None, max_rows: int = 5) -> dict:
    """Summarize the structure of an uploaded file without returning the full data."""
    path = _resolve_path(file_id)
    df, sheets, used_sheet = _read_table(path, sheet)

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    cols = [str(c) for c in df.columns]
    shown = cols[:_MAX_REPORT_COLS]

    dtypes = {c: ("numeric" if _is_numeric_col(df[orig]) else "text")
              for c, orig in zip(shown, list(df.columns)[:_MAX_REPORT_COLS])}

    stats = {}
    for c, orig in zip(shown, list(df.columns)[:_MAX_REPORT_COLS]):
        if dtypes[c] != "numeric":
            continue
        s = df[orig].dropna()
        if s.empty:
            continue
        stats[c] = {
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "mean": round(float(s.mean()), 6),
            "n_valid": int(s.size),
        }

    head_rows = []
    for _, row in df.head(max(1, min(int(max_rows or 5), 20))).iterrows():
        head_rows.append([_cell(row[orig]) for orig in list(df.columns)[:_MAX_REPORT_COLS]])

    out = {
        "ok": True,
        "file_id": file_id,
        "filename": path.name,
        "kind": _kind(path),
        "bytes": path.stat().st_size,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": shown,
        "dtypes": dtypes,
        "stats": stats,
        "head": head_rows,
        "note": (
            "This is a structural summary only. Decide yourself what each column means "
            "(e.g. whether a numeric column is Raman shift in cm-1, intensity, wavelength, "
            "a coordinate, or unrelated metadata) - nothing here has been interpreted for you. "
            "To compute on the full data, call run_analysis with file_ids=['" + str(file_id) + "']."
        ),
    }
    if sheets:
        out["sheets"] = sheets[:_MAX_REPORT_COLS]
        out["sheet"] = used_sheet
    if n_cols > _MAX_REPORT_COLS:
        out["columns_truncated"] = True
        out["note"] += f" (Only the first {_MAX_REPORT_COLS} of {n_cols} columns are listed here.)"
    return out


def load_upload(file_id: str, sheet: str | None = None) -> dict:
    """Load the full data of an uploaded file for sandbox injection."""
    path = _resolve_path(file_id)
    df, sheets, used_sheet = _read_table(path, sheet)

    truncated = False
    if df.shape[0] > _MAX_LOAD_ROWS:
        df = df.head(_MAX_LOAD_ROWS)
        truncated = True

    numeric: dict[str, list] = {}
    text: dict[str, list] = {}
    for orig in df.columns:
        name = str(orig)
        col = df[orig]
        if _is_numeric_col(col):
            numeric[name] = [None if v != v else float(v) for v in col.tolist()]
        else:
            text[name] = ["" if v != v else str(v) for v in col.tolist()]

    return {
        "file_id": file_id,
        "filename": path.name,
        "sheet": used_sheet,
        "sheets": sheets,
        "columns": [str(c) for c in df.columns],
        "n_rows": int(df.shape[0]),
        "numeric": numeric,
        "text": text,
        "truncated": truncated,
    }
