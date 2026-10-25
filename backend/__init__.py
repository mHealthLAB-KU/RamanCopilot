"""Backend package. Makes console output replace characters the console encoding cannot print."""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass
del _stream
