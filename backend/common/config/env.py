"""Helpers for reading environment variables shared by the config modules."""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv as _load_dotenv
    from pathlib import Path as _Path

    _load_dotenv(_Path(__file__).resolve().parents[3] / ".env")
except Exception:  # noqa: BLE001
    pass


def env(*names: str, default: str) -> str:
    """Return the first non-empty environment variable among names. Whitespace-only counts as unset."""
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return default


def flag(*names: str, default: str) -> bool:
    """On/off setting. Only '0' 'false' 'no' 'off' mean off; everything else means on."""
    return env(*names, default=default).lower() not in ("0", "false", "no", "off")
