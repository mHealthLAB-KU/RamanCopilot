"""SQLite store of measurement records and annotations. Rows are append-only."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from backend.common.config import virtual_hw as vhw_cfg
from backend.repo.store import DATA_ROOT

ROOT: Path = DATA_ROOT / "memory"


def root() -> Path:
    """Return the memory folder for the current hardware (a separate folder for virtual hardware)."""
    return ROOT / "virtual" if vhw_cfg.VIRTUAL_HW else ROOT


def db_path() -> Path:
    """Return the path of the records database."""
    return root() / "records.db"

MEAS_COLS = (
    "record_uid", "created_at",
    "session_id", "run_id", "skill_version", "tool_call_id",
    "power_mw", "exposure_s", "acq_mode", "num_accumulations", "kinetic_count", "n_frames",
    "shutter", "trigger_mode",
    "objective", "x_mm", "y_mm", "z_mm", "sample_name", "sample_source",
    "expected_bands", "expected_bands_source",
    "band_values", "spectrum_file",
    "image_before_file", "image_after_file", "lpips",
    "error_note",
    "spectrum_qc", "marker_bands", "purpose", "spot_label",
)
_JSON_COLS = ("expected_bands", "band_values", "spectrum_qc", "marker_bands")
_ADDED_COLS = {"spectrum_qc": "TEXT", "marker_bands": "TEXT", "purpose": "TEXT", "spot_label": "TEXT"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_uid TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
    session_id TEXT, run_id TEXT, skill_version INTEGER, tool_call_id TEXT NOT NULL,
    power_mw REAL, exposure_s REAL, acq_mode TEXT, num_accumulations INTEGER,
    kinetic_count INTEGER, n_frames INTEGER, shutter TEXT, trigger_mode TEXT,
    objective TEXT, x_mm REAL, y_mm REAL, z_mm REAL,
    sample_name TEXT, sample_source TEXT NOT NULL DEFAULT 'unknown',
    expected_bands TEXT, expected_bands_source TEXT,
    band_values TEXT, spectrum_file TEXT,
    image_before_file TEXT, image_after_file TEXT, lpips REAL,
    error_note TEXT
);
CREATE INDEX IF NOT EXISTS ix_meas_run ON measurement(run_id);
CREATE INDEX IF NOT EXISTS ix_meas_sample ON measurement(sample_name);
CREATE INDEX IF NOT EXISTS ix_meas_created ON measurement(created_at);

CREATE TABLE IF NOT EXISTS annotation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_uid TEXT NOT NULL, created_at TEXT NOT NULL,
    author TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ann_target ON annotation(target_uid);

CREATE TRIGGER IF NOT EXISTS measurement_no_update BEFORE UPDATE ON measurement
BEGIN SELECT RAISE(ABORT, 'append-only: measurement rows cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS measurement_no_delete BEFORE DELETE ON measurement
BEGIN SELECT RAISE(ABORT, 'append-only: measurement rows cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS annotation_no_update BEFORE UPDATE ON annotation
BEGIN SELECT RAISE(ABORT, 'append-only: annotations cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS annotation_no_delete BEFORE DELETE ON annotation
BEGIN SELECT RAISE(ABORT, 'append-only: annotations cannot be deleted'); END;
"""

ANNOTATION_KINDS = ("sample_identity", "quality", "error", "note")
ANNOTATION_AUTHORS = ("user", "agent")

_LOCK = threading.Lock()
_READY: set[str] = set()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """Open a connection, creating the schema on first use for each path."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA recursive_triggers = ON")
    key = str(path)
    if key not in _READY:
        con.executescript(_SCHEMA)
        have = {r[1] for r in con.execute("PRAGMA table_info(measurement)")}
        for col, typ in _ADDED_COLS.items():
            if col not in have:
                con.execute(f"ALTER TABLE measurement ADD COLUMN {col} {typ}")
        con.commit()
        _READY.add(key)
    return con


def insert_measurement(row: dict) -> str:
    """Insert one measurement row and return its record_uid."""
    unknown = set(row) - set(MEAS_COLS)
    if unknown:
        raise KeyError(f"unknown measurement columns: {sorted(unknown)}")
    vals = dict(row)
    vals.setdefault("record_uid", str(uuid.uuid4()))
    vals.setdefault("created_at", _now())
    for c in _JSON_COLS:
        if vals.get(c) is not None and not isinstance(vals[c], str):
            vals[c] = json.dumps(vals[c], ensure_ascii=False)
    cols = [c for c in MEAS_COLS if c in vals]
    with _LOCK, _connect() as con:
        con.execute(f"INSERT INTO measurement ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [vals[c] for c in cols])
    return vals["record_uid"]


def insert_annotation(target_uid: str, author: str, kind: str, body: dict) -> int:
    """Insert one annotation for an existing record."""
    if author not in ANNOTATION_AUTHORS:
        raise ValueError(f"author must be one of {ANNOTATION_AUTHORS}")
    if kind not in ANNOTATION_KINDS:
        raise ValueError(f"kind must be one of {ANNOTATION_KINDS}")
    with _LOCK, _connect() as con:
        if con.execute("SELECT 1 FROM measurement WHERE record_uid = ?", (target_uid,)).fetchone() is None:
            raise KeyError(f"no measurement {target_uid}")
        cur = con.execute("INSERT INTO annotation (target_uid, created_at, author, kind, body) "
                          "VALUES (?, ?, ?, ?, ?)",
                          (target_uid, _now(), author, kind, json.dumps(body, ensure_ascii=False)))
        return int(cur.lastrowid)


def resolve_uid(prefix: str) -> str:
    """Resolve a record id or its prefix (8+ characters) to the full id."""
    p = (prefix or "").strip()
    if len(p) < 8:
        raise KeyError("record id must be at least 8 characters")
    with _connect() as con:
        hits = [r[0] for r in con.execute(
            "SELECT record_uid FROM measurement WHERE record_uid LIKE ? LIMIT 2", (p + "%",))]
    if len(hits) != 1:
        raise KeyError(f"no measurement record '{p}'" if not hits else f"record id '{p}' is ambiguous")
    return hits[0]


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d.pop("id", None)
    for c in _JSON_COLS:
        if d.get(c):
            d[c] = json.loads(d[c])
    return d


def _attach_annotations(con, rows: list[dict]) -> None:
    """Attach the latest annotation of each kind and the corrected sample name to records."""
    if not rows:
        return
    uids = [r["record_uid"] for r in rows]
    latest: dict = {}
    q = (f"SELECT target_uid, kind, author, created_at, body FROM annotation "
         f"WHERE target_uid IN ({', '.join('?' * len(uids))}) ORDER BY id")
    for a in con.execute(q, uids):
        latest.setdefault(a["target_uid"], {})[a["kind"]] = {
            "author": a["author"], "created_at": a["created_at"], **json.loads(a["body"])}
    for r in rows:
        ann = latest.get(r["record_uid"], {})
        r["annotations"] = ann
        ident = ann.get("sample_identity") or {}
        r["effective_sample"] = ident.get("sample_name") or r.get("sample_name")


def measurements(*, run_id: str | None = None, uids=None, sample_name: str | None = None,
                 session_id: str | None = None, near: tuple | None = None,
                 since_days: float | None = None,
                 identity: str | None = None, limit: int = 50) -> list[dict]:
    """Return matching records (newest first) with their latest annotations."""
    where, args = [], []
    if identity:
        where.append("record_uid IN (SELECT target_uid FROM annotation WHERE kind = 'sample_identity' "
                     "AND lower(json_extract(body, '$.sample_name')) = lower(?))"); args.append(identity)
    if run_id is not None:
        where.append("run_id = ?"); args.append(run_id)
    if uids is not None:
        uids = list(uids)
        if not uids:
            return []
        where.append(f"record_uid IN ({', '.join('?' * len(uids))})"); args += uids
    if sample_name:
        where.append("(lower(sample_name) = lower(?) OR record_uid IN (SELECT target_uid FROM annotation "
                     "WHERE kind = 'sample_identity' AND lower(json_extract(body, '$.sample_name')) = lower(?)))")
        args += [sample_name, sample_name]
    if session_id is not None:
        where.append("session_id = ?"); args.append(session_id)
    if near is not None:
        x, y, r = near
        where.append("x_mm BETWEEN ? AND ? AND y_mm BETWEEN ? AND ?"); args += [x - r, x + r, y - r, y + r]
    if since_days is not None:
        where.append("created_at >= ?")
        args.append((datetime.now() - timedelta(days=since_days)).isoformat(timespec="seconds"))
    sql = "SELECT * FROM measurement"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit) * 4 if (near or sample_name or identity) else int(limit))
    with _connect() as con:
        rows = [_row(r) for r in con.execute(sql, args)]
        _attach_annotations(con, rows)
    if near is not None:
        x, y, r = near
        rows = [d for d in rows if (d["x_mm"] - x) ** 2 + (d["y_mm"] - y) ** 2 <= r * r]
    if sample_name:
        rows = [d for d in rows if (d["effective_sample"] or "").lower() == sample_name.lower()]
    if identity:
        rows = [d for d in rows if (identity_of(d) or "").lower() == identity.lower()]
    return rows[:int(limit)]


def acq_counts(rec: dict) -> tuple[int, int]:
    """Return (accumulations, kinetic_frames) of a record."""
    mode = rec.get("acq_mode")
    acc = int(rec.get("num_accumulations") or 1)
    if mode == "kinetic":
        return acc, int(rec.get("kinetic_count") or max(1, (rec.get("n_frames") or acc) // acc))
    if mode == "accumulate":
        return acc, 1
    return 1, 1


def identity_of(rec: dict) -> str | None:
    """Return the latest sample_identity annotation of a record, or None."""
    return ((rec.get("annotations") or {}).get("sample_identity") or {}).get("sample_name")
