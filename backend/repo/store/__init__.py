"""Disk storage under DATA and DETAILLOG."""


from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = _PROJECT_ROOT / "DATA"

LOG_ROOT = _PROJECT_ROOT / "DETAILLOG"
