"""Stores the measured laser spot position for each objective."""

from __future__ import annotations

import json
from datetime import datetime

from backend.repo.store import DATA_ROOT

AIM_FILE = DATA_ROOT / "calibration" / "laser_aim.json"


def _read_all() -> dict:
    try:
        return json.loads(AIM_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load(objective: str) -> tuple[float, float] | None:
    """Return the stored laser spot (x, y) for an objective, or None."""
    entry = _read_all().get(objective)
    if not entry:
        return None
    return (float(entry["x"]), float(entry["y"]))


def save(objective: str, x: float, y: float, **meta) -> None:
    """Store the laser spot for an objective together with measurement metadata."""
    data = _read_all()
    data[objective] = {"x": round(float(x), 2), "y": round(float(y), 2),
                       "measured_at": datetime.now().isoformat(timespec="seconds"), **meta}
    AIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    AIM_FILE.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
