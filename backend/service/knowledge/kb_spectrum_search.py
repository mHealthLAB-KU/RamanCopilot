#!/usr/bin/env python3
"""Search of the RRUFF library with a measured spectrum."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from backend.service.knowledge.calculate_spectrum_score import SpectrumMatcher

KB_SOURCES_DIR = Path(__file__).resolve().parent / "kb_sources"
DB_PATH = KB_SOURCES_DIR / "RamanLab_Database_20250602.pkl"

RERANK_N = 50  # candidates rescored after the correlation search

ALGORITHMS = {
    "correlation":       "calculate_correlation_score",
    "multi_window":      "calculate_multi_window_score",
    "mineral_vibration": "calculate_mineral_vibration_score",
    "dtw":               "calculate_dtw_score",
    "peak":              "calculate_peak_score",
    "combined":          "calculate_combined_score",
}

_CACHE: dict | None = None


def load_database(path=None):
    """Load the reference spectrum database.

    Args:
        path (str or Path, optional): Path to the pickle file. Defaults to DB_PATH.

    Returns:
        dict: The database as stored.
    """
    return pickle.loads(Path(path or DB_PATH).read_bytes())


_MARKUP = [
    (re.compile(r"&#183;|&middot;"), "·"),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"_([^_]*)_"), r"\1"),
    (re.compile(r"\^([^^]*)\^"), r"\1"),
    (re.compile(r"[\s\xa0]+"), " "),
]


def _clean(v) -> str:
    """Clean RRUFF metadata into a readable string."""
    s = str(v)
    for pattern, repl in _MARKUP:
        s = pattern.sub(repl, s)
    return s.strip()


def _cache() -> dict:
    """Build and keep the search cache."""
    global _CACHE
    if _CACHE is None:
        raw = load_database()
        _CACHE = {
            "spectra": {n: (np.asarray(e["wavenumbers"], np.float32),
                            np.asarray(e["intensities"], np.float32))
                        for n, e in raw.items()},
            "meta": {n: e.get("metadata") or {} for n, e in raw.items()},
        }
    return _CACHE


def reload_db() -> None:
    """Clear the cached database so the next search reloads it from disk."""
    global _CACHE
    _CACHE = None


def db_status() -> dict:
    """Report whether the reference library is loaded.

    Returns:
        dict: ok, number of entries, number of minerals and database path.
    """
    if not DB_PATH.exists():
        return {"ok": False, "entries": 0, "db_path": str(DB_PATH),
                "error": "RRUFF pkl not found"}
    c = _cache()
    return {"ok": True,
            "entries": len(c["spectra"]),
            "minerals": len({_mineral(n) for n in c["spectra"]}),
            "db_path": str(DB_PATH)}


def _mineral(name: str) -> str:
    return name.split("__")[0]


def _laser_nm(name: str):
    parts = name.split("__")
    return int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None


def identify(wavenumbers, intensities, top_k: int = 5, algorithm: str = "correlation",
             subtract_background: bool = True) -> list[dict]:
    """Return the RRUFF minerals most similar to a query spectrum.

    Args:
        wavenumbers (array-like): Raman shift axis of the query.
        intensities (array-like): Intensities of the query.
        top_k (int): Number of minerals to return.
        algorithm (str): Key of the scoring method in ALGORITHMS.
        subtract_background (bool): Whether to subtract an ALS background from the query first.

    Returns:
        list[dict]: One entry per mineral, sorted by score.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; one of {sorted(ALGORITHMS)}")

    wn = np.asarray(wavenumbers, dtype=float)
    y = np.asarray(intensities, dtype=float)
    if wn.size == 0 or wn.size != y.size:
        raise ValueError(f"wavenumbers/intensities length mismatch: {wn.size} vs {y.size}")

    if subtract_background:
        y, _ = subtract_background_als(wn, y)

    peaks = find_peaks(y, height=y.max() * 0.1)[0] if y.max() > 0 else np.array([], int)
    matcher = SpectrumMatcher(wn, y, peaks)

    cache = _cache()
    spectra = cache["spectra"]

    scores = {n: matcher.calculate_correlation_score(w, i) for n, (w, i) in spectra.items()}

    if algorithm != "correlation":
        score_of = getattr(matcher, ALGORITHMS[algorithm])
        shortlist = sorted(scores, key=scores.get, reverse=True)[:RERANK_N]
        scores = {n: score_of(*spectra[n]) for n in shortlist}

    best: dict[str, tuple[float, str]] = {}
    counts: dict[str, int] = {}
    for name, score in scores.items():
        m = _mineral(name)
        counts[m] = counts.get(m, 0) + 1
        if score > best.get(m, (-1.0, ""))[0]:
            best[m] = (score, name)

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

    out = []
    for mineral, (score, name) in ranked:
        md = cache["meta"].get(name, {})
        out.append({
            "mineral":           mineral,
            "score":             round(float(score), 4),
            "rruff_id":          _clean(md.get("RRUFF ID") or md.get("RRUFFID") or ""),
            "laser_nm":          _laser_nm(name),
            "formula":           _clean(md.get("IDEAL CHEMISTRY", "")),
            "elements":          _clean(md.get("CHEMISTRY ELEMENTS", "")),
            "crystal_system":    _clean(md.get("CRYSTAL SYSTEM", "")),
            "space_group":       _clean(md.get("SPACE GROUPS", "")),
            "classification":    _clean(md.get("HEY CLASSIFICATION", "")),
            "n_library_spectra": counts[mineral],
            "entry":             name,
        })
    return out


def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """Estimate a baseline by asymmetric least squares smoothing.

    Args:
        y (array-like): Input spectrum.
        lam (float): Smoothness parameter.
        p (float): Asymmetry parameter.
        niter (int): Number of iterations.

    Returns:
        np.ndarray: Estimated baseline.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc")
    lam_DTD = lam * (D.T @ D)
    w = np.ones(L)

    for i in range(niter):
        z = spsolve(diags(w, format="csc") + lam_DTD, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)

    return z


def subtract_background_als(wavenumbers, intensities, lam=1e5, p=0.01, niter=10):
    """Subtract an ALS baseline from a spectrum.

    Args:
        wavenumbers (array-like): Raman shift axis. Unused, kept for compatibility with RamanLab.
        intensities (array-like): Intensities.
        lam (float): Smoothness parameter.
        p (float): Asymmetry parameter.
        niter (int): Number of iterations.

    Returns:
        tuple[np.ndarray, np.ndarray]: Corrected intensities and baseline.
    """
    intensities = np.asarray(intensities, dtype=float)

    baseline = baseline_als(intensities, lam, p, niter)

    corrected_spectrum = intensities - baseline

    corrected_spectrum[corrected_spectrum < 0] = 0

    return corrected_spectrum, baseline
