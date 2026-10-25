"""Baseline and background subtraction shared by tools and the sandbox."""

from __future__ import annotations

import numpy as np


def poly_baseline(y, order: int = 5, x=None):
    """Fit a polynomial baseline once."""
    y = np.asarray(y, dtype=np.float64)
    xx = np.linspace(0.0, 1.0, len(y)) if x is None else np.asarray(x, dtype=np.float64)
    return np.polyval(np.polyfit(xx, y, deg=int(order)), xx)


def ipbsa_detail(y, order: int = 5, max_iterations: int = 100,
                 threshold: float = 0.001, x=None):
    """Iterative polynomial baseline subtraction; return (corrected, baseline, iterations, converged)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    xx = np.linspace(0.0, 1.0, n) if x is None else np.asarray(x, dtype=np.float64)
    working = y.copy()
    prev_bg = np.zeros(n, dtype=np.float64)
    bg = prev_bg
    converged = False
    iters = max(1, int(max_iterations))
    i = 0
    for i in range(iters):
        bg = np.polyval(np.polyfit(xx, working, deg=int(order)), xx)
        working = np.minimum(y, bg)
        denom = np.linalg.norm(prev_bg)
        if denom > 0 and np.linalg.norm(bg - prev_bg) / denom < threshold:
            converged = True
            break
        prev_bg = bg.copy()
    return np.clip(y - bg, 0.0, None), bg, i + 1, converged


def ipbsa(y, order: int = 5, max_iterations: int = 100,
          threshold: float = 0.001, x=None):
    """Iterative polynomial baseline subtraction; return (corrected, baseline)."""
    corrected, bg, _, _ = ipbsa_detail(y, order=order, max_iterations=max_iterations,
                                       threshold=threshold, x=x)
    return corrected, bg
