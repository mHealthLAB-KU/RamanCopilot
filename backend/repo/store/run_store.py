"""Per-session storage of agent outputs."""

from __future__ import annotations

import json
import re
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.repo.store import DATA_ROOT
from backend.repo.store.paths import RUNS_ROOT

KIND_SPECTRA = "spectra"
KIND_FIGURE = "figure"
KIND_MEASUREMENT = "measurement"
_DIR_KINDS = {KIND_SPECTRA}

_LOCK = threading.Lock()

class _Current(threading.local):
    def __init__(self):
        self.session_id = ""
        self.agent = ""
        self.label = ""
        self.isolated = False


_current = _Current()


def sanitize(session_id: str) -> str:
    """Convert a session id into a filesystem-safe label (max 64 characters)."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(session_id))[:64] or "nosession"


def new_session_id() -> str:
    """Return a new session id for chat sessions."""
    from backend.common.config import virtual_hw as vhw
    if not vhw.VIRTUAL_HW:
        return str(uuid.uuid4())
    base = f"vbench_{datetime.now():%Y%m%d_%H%M%S}"
    sid, n = base, 1
    while (RUNS_ROOT / sid).exists():
        n += 1
        sid = f"{base}_{n}"
    return sid


def log_stem(agent: str, stamp: str, session_id: str) -> str:
    """Return the turn-log file stem for a session."""
    return f"{agent}_{stamp}_{sanitize(session_id)}"


def existing_log(log_dir: Path, agent: str, session_id: str, suffix: str) -> Optional[Path]:
    """Return this session's existing turn log on disk, or None."""
    sid = sanitize(session_id)
    if not sid:
        return None
    hits = sorted(log_dir.glob(f"{agent}_*_{sid}{suffix}"))
    return hits[0] if hits else None


def _safe_name(name, fallback: str = "result") -> str:
    """Reduce a model-provided artifact name to a safe single-part stem."""
    raw = str(name or "").strip()
    tail = raw.replace("\\", "/").rstrip("/").split("/")[-1]
    tail = tail.replace(":", "_")
    if tail.lower().endswith(".csv"):
        tail = tail[:-4]
    safe = "".join(c for c in tail if c.isalnum() or c in "._- ").strip(" .")
    safe = re.sub(r"\s+", "_", safe)
    return safe[:60] or fallback


def begin_session(session_id: str, agent: str = "", isolated: bool = False) -> dict:
    """Start (or resume) a session at the beginning of a turn."""
    label = sanitize(session_id)
    _current.session_id = str(session_id or "")
    _current.agent = str(agent or "")
    _current.label = label
    _current.isolated = bool(isolated)
    try:
        d = RUNS_ROOT / label
        d.mkdir(parents=True, exist_ok=True)
        mpath = d / "manifest.json"
        if not mpath.exists():
            from backend.common.config import virtual_hw as vhw
            _write_manifest({
                "session_id": _current.session_id,
                "agent": _current.agent,
                "label": label,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "virtual": vhw.VIRTUAL_HW,
                "scene": vhw.scene_label(),
                "artifacts": [],
            })
    except Exception as e:
        print(f"[warn] run_store.begin_session: {type(e).__name__}: {e}", file=sys.stderr)
    return current()


def current() -> dict:
    """Return the current session information."""
    label = _current.label
    return {
        "session_id": _current.session_id,
        "agent": _current.agent,
        "label": label,
        "dir": str(RUNS_ROOT / label) if label else "",
        "rel_dir": f"runs/{label}" if label else "",
    }


def isolated_label() -> str | None:
    """Return the current session label in isolated mode, else None."""
    return _current.label if _current.isolated and _current.label else None


def session_dir() -> Path:
    """Return the current session directory ('_unassigned' if no session is open)."""
    label = _current.label or "_unassigned"
    d = RUNS_ROOT / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return session_dir() / "manifest.json"


def _write_manifest(doc: dict) -> None:
    _manifest_path().write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def manifest() -> dict:
    """Return the manifest of the current session."""
    try:
        p = _manifest_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] run_store.manifest: {type(e).__name__}: {e}", file=sys.stderr)
    return {"session_id": _current.session_id, "label": _current.label, "artifacts": []}


def record(kind: str, rel_path: str, **meta) -> dict:
    """Index one artifact in the session manifest."""
    entry = {"kind": kind, "path": rel_path,
             "saved_at": datetime.now().strftime("%H:%M:%S"), **meta}
    try:
        with _LOCK:
            doc = manifest()
            arts = [a for a in doc.get("artifacts", []) if a.get("path") != rel_path]
            arts.append(entry)
            doc["artifacts"] = arts
            doc.setdefault("session_id", _current.session_id)
            doc.setdefault("label", _current.label)
            doc["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _write_manifest(doc)
    except Exception as e:
        print(f"[warn] run_store.record: {type(e).__name__}: {e}", file=sys.stderr)
    return entry


def next_index(kind: str) -> int:
    """Return the next sequence number for an artifact kind."""
    return sum(1 for a in manifest().get("artifacts", []) if a.get("kind") == kind) + 1


def new_artifact_path(kind: str, name, ext: str = ".csv") -> tuple[Path, str]:
    """Return a numbered save path for an artifact kind as (absolute path, path relative to DATA)."""
    if kind not in _DIR_KINDS:
        raise ValueError(f"kind must be one of {sorted(_DIR_KINDS)} (got {kind!r})")
    d = session_dir() / kind
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{next_index(kind):02d}_{_safe_name(name)}"
    p = d / f"{stem}{ext}"
    n = 2
    while p.exists():
        p = d / f"{stem}-{n}{ext}"
        n += 1
    label = _current.label or "_unassigned"
    return p, f"runs/{label}/{kind}/{p.name}"


def new_spectrum_path(name, ext: str = ".csv") -> tuple[Path, str]:
    """Return a save path for a processed spectrum."""
    return new_artifact_path(KIND_SPECTRA, name, ext)


def list_artifacts(kind: str | None = None) -> list[dict]:
    """Return the artifacts of the current session, optionally filtered by kind."""
    arts = manifest().get("artifacts", [])
    if kind:
        arts = [a for a in arts if a.get("kind") == kind]
    return arts


def summary_for_prompt() -> str:
    """Return a short session summary for the system prompt."""
    cur = current()
    if not cur["label"]:
        return ""
    arts = list_artifacts()
    head = (f"Your session label is `{cur['label']}`. Files you save go under "
            f"`{cur['rel_dir']}/`, and spectra you measure are auto-saved under "
            f"`results/<date>/{cur['label']}/`. Both are listed below and both are "
            f"readable with open_file('<path>').")
    if isolated_label():
        head += (" This run is ISOLATED: you can only read files from your own session "
                 "(and files the user attached). Measurements from other sessions are not "
                 "available - if you need data, measure it yourself in this run.")
    if not arts:
        return head + " You have not saved any artifact in this session yet."
    lines = [f"  - [{a.get('kind')}] {a.get('path')}"
             + (f"  ({a.get('num_points')} pts)" if a.get("num_points") else "")
             for a in arts[-12:]]
    more = "" if len(arts) <= 12 else f"  ... and {len(arts) - 12} earlier artifact(s)\n"
    return (head + f" Artifacts you have already saved in this session ({len(arts)}):\n"
            + more + "\n".join(lines)
            + "\nRead any of these back with open_file('<path>') or run_analysis.")
