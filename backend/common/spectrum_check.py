"""Saturation check for measured spectra."""
from __future__ import annotations

import numpy as np

FULL_WELL = 65535  # 16-bit ADC full well, counts
SAT_FRACTION = 0.95  # fraction of FULL_WELL treated as saturated
SAT_COUNTS = FULL_WELL * SAT_FRACTION


def saturated(y) -> bool:
    """Check whether a spectrum reaches the saturation threshold.

    Args:
        y (array-like): Intensities.

    Returns:
        bool: True if the maximum is at or above the threshold. False for empty input.
    """
    a = np.asarray(y, dtype=float)
    return bool(a.size) and float(np.nanmax(a)) >= SAT_COUNTS
