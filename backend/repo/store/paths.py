"""Resolves file identifiers to paths on disk."""

from __future__ import annotations

import re
from pathlib import Path

from backend.repo.store import DATA_ROOT

UPLOADS_ROOT = DATA_ROOT / "uploads"
RESULTS_ROOT = DATA_ROOT / "results"
RUNS_ROOT = DATA_ROOT / "runs"

KB_ROOT = Path(__file__).resolve().parents[2] / "service" / "knowledge" / "kb_sources"

ROOTS: dict[str, Path] = {
    "uploads": UPLOADS_ROOT,
    "results": RESULTS_ROOT,
    "runs": RUNS_ROOT,
    "kb": KB_ROOT,
}

URL_PREFIX = "/api/results"

KIND_SUFFIXES: dict[str, set[str]] = {
    "image": {".png", ".jpg", ".jpeg"},
    "table": {".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls"},
    "json":  {".json"},
    "text":  {".md"},
}

MEMORY_PREFIX = "memory"
LATEST_SKILL = "latest.md"


def _skill_dir() -> Path:
    from backend.repo.store import skill_store
    return skill_store.directory()


KIND_TOOL = {k: "open_file" for k in KIND_SUFFIXES}


def kind_of(path: Path | str) -> str | None:
    """Return the file kind for an extension, or None."""
    suf = Path(path).suffix.lower()
    return next((k for k, sufs in KIND_SUFFIXES.items() if suf in sufs), None)

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class FileIdError(ValueError):
    """Raised when a file_id cannot be resolved; the message is read by the model."""


def make_id(root: str, *parts: str) -> str:
    """Build a file_id from a root and relative path parts."""
    if root not in ROOTS:
        raise ValueError(f"Unknown root {root!r} (expected one of {sorted(ROOTS)})")
    rel = "/".join(str(p).strip("/\\") for p in parts if str(p).strip("/\\"))
    return f"{root}:{rel}"


def split_id(file_id: str) -> tuple[str | None, str]:
    """Split a file_id into (root, relative path)."""
    raw = str(file_id or "").strip().replace("\\", "/")
    head, sep, tail = raw.partition(":")
    if sep and (head.strip().lower() in ROOTS or head.strip().lower() == MEMORY_PREFIX):
        return head.strip().lower(), tail.lstrip("/")
    return None, raw


def _safe_parts(rel: str) -> list[str]:
    """Split a relative path into parts, rejecting path traversal."""
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p != Path(p).name for p in parts):
        raise FileIdError(f"Invalid file_id: {rel!r} (it must be a plain relative path)")
    return parts


def _known_suffixes() -> str:
    return ", ".join(sorted({s for sufs in KIND_SUFFIXES.values() for s in sufs}))


def _id_of(path: Path) -> str:
    """Return the canonical file_id for a real path."""
    for name, root in ROOTS.items():
        try:
            return f"{name}:{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return str(path)


def _check_kind(path: Path, kind: str | None, file_id: str) -> Path:
    """Check that the file extension matches the requested kind."""
    if kind is None:
        return path
    allowed = KIND_SUFFIXES.get(kind)
    if allowed is None:
        raise ValueError(f"Unknown kind {kind!r} (expected one of {sorted(KIND_SUFFIXES)})")
    if path.suffix.lower() in allowed:
        return path
    raise FileIdError(
        f"'{file_id}' is a {path.suffix or 'no-extension'} file, but a "
        f"{kind} file was required here (accepted: {', '.join(sorted(allowed))}).")


def _owner_session(path: Path) -> tuple[str, str | None]:
    """Return (root name, owning session label) for a path."""
    try:
        path.relative_to(_skill_dir())
        return MEMORY_PREFIX, None
    except Exception:  # noqa: BLE001
        pass
    for name, root in ROOTS.items():
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if name in ("uploads", "kb"):
            return name, None
        if name == "runs":
            return name, parts[0] if parts else ""
        return name, parts[1] if len(parts) >= 3 else ""
    return "", ""


def _accept(path: Path, kind: str | None, file_id: str) -> Path:
    """Final check of resolve(): extension and session isolation."""
    path = _check_kind(path, kind, file_id)

    from backend.repo.store import run_store
    mine = run_store.isolated_label()
    if mine is None:
        return path

    root, owner = _owner_session(path)
    if owner is None:
        return path
    if owner == mine:
        return path
    if not root:
        raise FileIdError(
            f"'{file_id}' points outside the managed data folders, and this run may only read "
            f"its own session's files. Use an id from list_results(), list_session_artifacts() "
            f"or list_uploaded_files().")
    raise FileIdError(
        f"'{file_id}' belongs to "
        + (f"another session ({owner})" if owner else "no session (an older file)")
        + f", and this run may only read its own session `{mine}`. "
          f"List your own files with list_results() or list_session_artifacts(), "
          f"or measure it yourself in this run.")


def _skill_path(rel: str) -> Path:
    """Resolve 'memory:<name>' to a skill file path."""
    parts = _safe_parts(rel)
    if len(parts) != 1:
        raise FileIdError(f"Invalid skill id: {rel!r}. Use 'memory:latest.md' (the current skills) "
                          f"or 'memory:v0001.md' (a specific version).")
    name = parts[0]
    if name.lower() == LATEST_SKILL:
        from backend.repo.store import skill_store
        v = skill_store.latest_version()
        if v <= 0:
            raise FileIdError("No skill exists yet. Skills are written by reflection after a task that "
                              "took measurements, so there is nothing to read.")
        name = f"v{v:04d}.md"
    path = _skill_dir() / name
    if not path.is_file():
        raise FileIdError(f"Not found under 'memory': {parts[0]}. Skill files are named v0001.md, "
                          f"v0002.md, ... - 'memory:latest.md' is the current one.")
    return path


def resolve(file_id: str, kind: str | None = None) -> Path:
    """Resolve a file_id to an existing path (read only)."""
    raw = str(file_id or "").strip()
    if not raw:
        raise FileIdError("file_id is empty. Get one from list_uploaded_files, "
                          "list_results or an earlier tool result.")

    if raw.replace("\\", "/").startswith(URL_PREFIX):
        raw = "results:" + raw.replace("\\", "/")[len(URL_PREFIX):].lstrip("/")

    root, rel = split_id(raw)

    if root is None and Path(rel).is_absolute():
        p = Path(rel)
        if not p.is_file():
            raise FileIdError(f"File not found: {p}")
        return _accept(p, kind, file_id)

    if root == MEMORY_PREFIX:
        return _accept(_skill_path(rel), kind, file_id)

    if root is not None:
        path = ROOTS[root].joinpath(*_safe_parts(rel))
        if not path.is_file():
            raise FileIdError(
                f"Not found under '{root}': {rel}. The id names the {root} area but no such "
                f"file is there. List what exists with "
                f"{'list_uploaded_files' if root == 'uploads' else 'list_results' if root == 'results' else 'list_session_artifacts'}().")
        return _accept(path, kind, file_id)

    parts = _safe_parts(rel)
    local = DATA_ROOT.joinpath(*parts)
    if local.is_file():
        return _accept(local, kind, file_id)

    for name, root_dir in ROOTS.items():
        cand = root_dir.joinpath(*parts)
        if cand.is_file():
            return _accept(cand, kind, file_id)

    if len(parts) == 1:
        for name, root_dir in ROOTS.items():
            if not root_dir.is_dir():
                continue
            for day in sorted((d for d in root_dir.iterdir() if d.is_dir()), reverse=True):
                cand = day / parts[0]
                if cand.is_file():
                    return _accept(cand, kind, file_id)

    raise FileIdError(
        f"No file matches the id {file_id!r}. Looked under data/ and in "
        f"{', '.join(ROOTS)}. Ids look like 'uploads:2026-08-12/N05.csv' or "
        f"'results:2026-08-12/_microscope_1401.png' - get an exact one from "
        f"list_uploaded_files(), list_results() or list_session_artifacts().")


def try_resolve(file_id: str, kind: str | None = None) -> tuple[Path | None, str]:
    """Exception-free resolve(); return (path, error)."""
    try:
        return resolve(file_id, kind), ""
    except (FileIdError, ValueError) as e:
        return None, str(e)


__all__ = ["DATA_ROOT", "UPLOADS_ROOT", "RESULTS_ROOT", "RUNS_ROOT", "KB_ROOT", "ROOTS", "URL_PREFIX",
           "MEMORY_PREFIX", "LATEST_SKILL",
           "KIND_SUFFIXES", "KIND_TOOL", "FileIdError",
           "make_id", "split_id", "kind_of", "resolve", "try_resolve"]
