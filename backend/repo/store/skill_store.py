"""Versioned storage of skill files written by reflection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from backend.repo.store import memory_db

SKILL_DIR: Path | None = None


def _dir() -> Path:
    return SKILL_DIR if SKILL_DIR is not None else memory_db.root() / "skills"

def directory() -> Path:
    """Return the skill folder."""
    return _dir()


_NAME = re.compile(r"^v(\d{4,})\.md$")


def _versions() -> list[int]:
    d = _dir()
    if not d.exists():
        return []
    return sorted(int(m.group(1)) for p in d.iterdir() if (m := _NAME.match(p.name)))


def latest_version() -> int:
    """Return the highest skill version number (0 if none)."""
    v = _versions()
    return v[-1] if v else 0


def read(version: int) -> tuple[dict, list[dict]]:
    """Return (metadata, rules) of a skill version."""
    if version <= 0:
        return {}, []
    text = (_dir() / f"v{version:04d}.md").read_text(encoding="utf-8")
    _, head, body = text.split("---", 2)
    return yaml.safe_load(head) or {}, yaml.safe_load(body) or []


def write(rules: list[dict], meta: dict) -> int:
    """Write a new skill version and return its number."""
    _dir().mkdir(parents=True, exist_ok=True)
    parent = latest_version()
    version = parent + 1
    head = {"version": version, "parent": parent or None, **meta}
    text = ("---\n" + yaml.safe_dump(head, allow_unicode=True, sort_keys=False) + "---\n"
            + yaml.safe_dump([canonical(r) for r in rules], allow_unicode=True, sort_keys=False))
    with open(_dir() / f"v{version:04d}.md", "x", encoding="utf-8") as f:
        f.write(text)
    return version


def canonical(rule: dict) -> dict:
    """Return the canonical form of a rule for comparison and storage."""
    out = {"rule": str(rule.get("rule", "")).strip(),
           "scope": rule.get("scope") or {},
           "evidence": sorted({str(e) for e in rule.get("evidence") or []})}
    if rule.get("status") in ("candidate", "active"):
        out["status"] = rule["status"]
    return out


def same(a: list[dict], b: list[dict]) -> bool:
    """Return whether two rule snapshots are equivalent."""
    key = lambda r: (r["rule"], json.dumps(r["scope"], sort_keys=True, ensure_ascii=False), r["evidence"])
    return sorted(map(key, map(canonical, a))) == sorted(map(key, map(canonical, b)))


def render(version: int, rules: list[dict]) -> str:
    """Render the skill for the prompt."""
    lines = []
    for r in rules:
        sc = r.get("scope")
        scope = (", ".join(f"{k}={v}" for k, v in sc.items()) if isinstance(sc, dict) else str(sc or ""))             or "scope unspecified"
        ev = ", ".join(str(e)[:8] for e in r.get("evidence") or [])
        tag = " [CANDIDATE - one task only, not yet confirmed]" if r.get("status") == "candidate" else ""
        lines.append(f"- [{scope}]{tag} {r.get('rule', '')} (evidence: {ev})")
    return "\n".join(lines)
