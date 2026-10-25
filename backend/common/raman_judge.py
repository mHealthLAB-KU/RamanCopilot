"""Raman spectrum judging with silicon as the internal standard: axis correction, band detection and before/after comparison."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

SI_CM1 = 520.7  # Si first-order Raman peak, reference for axis correction
SI_MASK_CM1 = 16.0  # half-width around the Si peak that is not judged
MAX_AXIS_SHIFT_CM1 = 20.0  # larger shifts are not treated as Si
DETECT_SNR = 6.0  # minimum band height over smoothed noise
PROMINENCE_SIGMA = 4.0  # minimum height over the flanks, in noise sigma
SI_SYSTEMATIC = 0.5  # fraction of the subtracted Si reference counted as systematic error
FLANK_PERCENTILE = 25  # percentile used as the flank level
JUDGE_VERSION = "v2-si-ref-2026-10-19"

_SI_REF = Path(__file__).with_name("data") / "si_reference.csv"


@lru_cache(maxsize=1)
def si_reference() -> tuple[np.ndarray, np.ndarray]:
    """Load the bare silicon reference spectrum.

    Returns:
        tuple[np.ndarray, np.ndarray]: Raman shift axis aligned to Si 520.7 cm-1, and intensity normalized to a first-order peak height of 1.
    """
    d = np.loadtxt(_SI_REF, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def _mad(v) -> float:
    v = np.asarray(v, float)
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def despike(y, axis=None) -> np.ndarray:
    """Remove cosmic-ray spikes while keeping the silicon first-order peak.

    Args:
        y (array-like): Intensities.
        axis (array-like, optional): Raman shift axis. If given, the 480-560 cm-1 range is left unchanged.

    Returns:
        np.ndarray: Intensities with spikes replaced.
    """
    from scipy.ndimage import median_filter, uniform_filter1d
    y = np.asarray(y, float).copy()
    ref = median_filter(y, 11, mode="nearest")
    e = y - ref
    s0 = max(_mad(y - uniform_filter1d(y, 5, mode="nearest")), 1e-9)
    n = len(y)
    for i in np.flatnonzero(e > 6 * s0):
        half = e[i] / 2
        j0 = j1 = i
        while j0 > 0 and e[j0 - 1] > half:
            j0 -= 1
        while j1 < n - 1 and e[j1 + 1] > half:
            j1 += 1
        if j1 - j0 + 1 > 2:
            continue
        shoulder = max(e[j0 - 1] if j0 > 0 else 0.0, e[j1 + 1] if j1 < n - 1 else 0.0)
        if shoulder >= 0.3 * e[i]:
            continue
        if axis is not None and 480 <= axis[i] <= 560:
            continue
        y[j0:j1 + 1] = ref[j0:j1 + 1]
    return y


def point_noise(y) -> float:
    """Estimate the white-noise sigma of a single point.

    Args:
        y (array-like): Intensities.

    Returns:
        float: Noise sigma from the MAD of the 5-point moving-average residual.
    """
    from scipy.ndimage import uniform_filter1d
    return _mad(y - uniform_filter1d(y, 5, mode="nearest")) / np.sqrt(0.8)


def si_peak(axis, y, sigma: Optional[float] = None) -> Optional[dict]:
    """Locate the silicon first-order peak.

    Args:
        axis (array-like): Raman shift axis as reported by the instrument, before correction.
        y (array-like): Despiked intensities.
        sigma (float, optional): Point noise. Estimated from y if omitted.

    Returns:
        dict or None: si_peak_cm1, si_height, si_fwhm_cm1 and axis_shift_cm1, or None if no peak is found.
    """
    axis = np.asarray(axis, float)
    y = np.asarray(y, float)
    sigma = point_noise(y) if sigma is None else sigma
    m = (axis > 490) & (axis < 552)
    if m.sum() < 5:
        return None
    idx = np.flatnonzero(m)
    k = int(idx[np.argmax(y[idx])])
    if k <= 0 or k >= len(y) - 1:
        return None
    side = np.r_[y[(axis > 440) & (axis < 470)], y[(axis > 570) & (axis < 600)]]
    base = float(np.median(side)) if side.size else float(np.min(y[idx]))
    h = float(y[k] - base)
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den != 0 else 0.0
    pos = float(axis[k] + d * (axis[k + 1] - axis[k]))
    half = base + h / 2
    j0 = j1 = k
    while j0 > 0 and y[j0] > half:
        j0 -= 1
    while j1 < len(y) - 1 and y[j1] > half:
        j1 += 1
    fwhm = float(axis[j1] - axis[j0])
    if h < 10 * sigma or fwhm > 20:
        return None
    return {"si_peak_cm1": round(pos, 2), "si_height": round(h, 1), "si_fwhm_cm1": round(fwhm, 1),
            "axis_shift_cm1": round(SI_CM1 - pos, 2)}


def _baseline(y, win_pts: float) -> np.ndarray:
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    w = int(max(15, win_pts)) | 1
    return uniform_filter1d(minimum_filter1d(uniform_filter1d(y, 15, mode="nearest"), w, mode="nearest"),
                            w, mode="nearest")


def _prepare(axis, y):
    """Despike, estimate noise, calibrate the axis on Si and subtract the Si reference."""
    axis = np.asarray(axis, float)
    y = despike(y, axis)
    sigma = point_noise(y)
    si = si_peak(axis, y, sigma)
    shift = si["axis_shift_cm1"] if si and abs(si["axis_shift_cm1"]) <= MAX_AXIS_SHIFT_CM1 else 0.0
    xa = axis + shift
    tsi = np.zeros_like(y)
    if si:
        tx, ty = si_reference()
        tsi = si["si_height"] * np.interp(xa, tx, ty, left=0.0, right=0.0)
    mask = (np.abs(xa - SI_CM1) <= SI_MASK_CM1) if si else np.zeros_like(xa, bool)
    return xa, y, y - tsi, tsi, mask, sigma, si, shift


def spectrum_qc(axis, y) -> dict:
    """Compute quality metrics of one spectrum.

    Args:
        axis (array-like): Raman shift axis.
        y (array-like): Intensities.

    Returns:
        dict: Silicon peak values, axis shift, noise sigma, background level and saturation flag.
    """
    from backend.common import spectrum_check
    axis = np.asarray(axis, float)
    yr = np.asarray(y, float)
    out = {"judge_version": JUDGE_VERSION, "saturated": spectrum_check.saturated(yr)}
    xa, yd, ys, _, _, sigma, si, shift = _prepare(axis, yr)
    floor = float(np.percentile(yd, 1))
    base = _baseline(ys, 61)
    w = (xa > 1000) & (xa < 1800)
    out.update(si or {"si_peak_cm1": None, "si_height": None})
    out.update(axis_shift_cm1=round(float(shift), 2), noise_sigma=round(float(sigma), 2),
               background_counts=round(float(np.median(base[w]) - floor), 1) if w.any() else None)
    return out


def judge_bands(axis, y, bands: list[dict]) -> list[dict]:
    """Decide for each expected band whether it is detected.

    The silicon reference is subtracted and the axis is shifted onto Si 520.7 cm-1 before judging.

    Args:
        axis (array-like): Raman shift axis.
        y (array-like): Intensities.
        bands (list[dict]): Expected bands with center_cm1 and optional half_width_cm1, fwhm_cm1 and role.

    Returns:
        list[dict]: One entry per band with found_at_cm1, net_height, snr, detected and reason. detected is None when the band cannot be judged.
    """
    from scipy.ndimage import uniform_filter1d

    from backend.common import spectrum_check
    axis = np.asarray(axis, float)
    yr = np.asarray(y, float)
    xa, _, ys, tsi, simask, sigma, _, _ = _prepare(axis, yr)
    dx = float(np.median(np.diff(xa))) if len(xa) > 1 else 1.0
    sat = spectrum_check.saturated(yr)
    out = []
    for b in bands:
        c = float(b["center_cm1"])
        t = float(b.get("half_width_cm1") or 10.0)
        f = float(b.get("fwhm_cm1") or 2 * t)
        r = {"center_cm1": c, "half_width_cm1": t, "fwhm_cm1": f}
        if b.get("role"):
            r["role"] = b["role"]
        win = np.abs(xa - c) <= t
        if not win.any():
            out.append({"center_cm1": c, "in_range": False})
            continue
        if sat:
            r.update(net_height=None, snr=None, detected=None, saturated=True)
            out.append(r)
            continue
        if (win & simask).any():
            r.update(si_overlap=True, detected=None,
                     reason="window overlaps the Si 520.7 cm-1 line, so this band cannot be judged here")
            out.append(r)
            continue
        k_s = int(np.clip(round(f / dx / 3), 1, 11)) | 1
        yk = uniform_filter1d(ys, k_s, mode="nearest") if k_s > 1 else ys
        net = yk - _baseline(ys, (3 * f + 2 * t) / dx)
        lag = max(k_s, 2)
        sig_k = max(_mad(yk[lag:] - yk[:-lag]) / np.sqrt(2), sigma / np.sqrt(k_s))
        idx = np.flatnonzero(win)
        kk = int(idx[np.argmax(net[idx])])
        h = float(net[kk])
        inside = kk not in (int(idx[0]), int(idx[-1]))

        def flank(sign):
            d = sign * (xa - c)
            j = np.flatnonzero((d >= t + f / 2) & (d <= t + 2 * f) & ~simask)
            return float(np.percentile(net[j], FLANK_PERCENTILE)) if j.size else None
        fl = [v for v in (flank(-1), flank(+1)) if v is not None]
        edge = max(fl) if fl else 0.0
        sig_t = float(np.hypot(sig_k, SI_SYSTEMATIC * tsi[kk]))
        snr = h / sig_t if sig_t > 0 else float("inf")
        prom = (h - edge) / sig_t if sig_t > 0 else float("inf")
        r.update(found_at_cm1=round(float(xa[kk]), 1), net_height=round(h, 2), noise=round(sig_t, 2),
                 snr=round(float(snr), 2), prominence_snr=round(float(prom), 2), peak_inside=bool(inside),
                 si_overlap=False,
                 detected=bool(inside and prom >= PROMINENCE_SIGMA and snr >= DETECT_SNR))
        if not inside and snr >= DETECT_SNR:
            r["warning"] = (f"band top is at the edge of the search range ({c - t:g}-{c + t:g} cm-1, found at "
                            f"{xa[kk]:.1f}); the real peak likely lies outside it, so detected=false here may be wrong. "
                            f"Widen band_half_width (or recenter) in set_sample_context so the whole band fits.")
        out.append(r)
    return out


def _residual_snr(axis, y):
    """Return the calibrated axis, the Si- and baseline-subtracted intensity, and its SNR."""
    from scipy.ndimage import uniform_filter1d
    xa, _, ys, tsi, simask, sigma, _, _ = _prepare(axis, y)
    dx = float(np.median(np.diff(xa)))
    yk = uniform_filter1d(ys, 3, mode="nearest")
    net = yk - _baseline(ys, 120 / dx)
    noise = np.hypot(max(_mad(yk[3:] - yk[:-3]) / np.sqrt(2), sigma / np.sqrt(3)), SI_SYSTEMATIC * tsi)
    snr = np.where(simask, 0.0, net / noise)
    return xa, net, snr


def residual_peaks(axis, y, min_snr: float = 8.0) -> list[dict]:
    """List peaks that remain after subtracting the silicon reference.

    Args:
        axis (array-like): Raman shift axis.
        y (array-like): Intensities.
        min_snr (float): Minimum signal-to-noise ratio of a reported peak.

    Returns:
        list[dict]: Peaks with at_cm1, net_height and snr.
    """
    from scipy.signal import find_peaks
    xa, net, snr = _residual_snr(axis, y)
    dx = float(np.median(np.diff(xa)))
    pk, _ = find_peaks(snr, height=min_snr, prominence=min_snr / 2, width=(1, 100 / dx))
    return [{"at_cm1": round(float(xa[i]), 1), "net_height": round(float(net[i]), 1), "snr": round(float(snr[i]), 1)}
            for i in pk if xa[i] > 180]


def compare_spectra(axis_a, y_a, axis_b, y_b, bands: list[dict], scale_b: float = 1.0) -> dict:
    """Compare two probe spectra taken at the same spot before and after irradiation.

    Args:
        axis_a (array-like): Raman shift axis of the spectrum before irradiation.
        y_a (array-like): Intensities before irradiation.
        axis_b (array-like): Raman shift axis of the spectrum after irradiation.
        y_b (array-like): Intensities after irradiation.
        bands (list[dict]): Expected bands.
        scale_b (float): Factor that brings b to the power and exposure of a.

    Returns:
        dict: Band ratios and significance, new peaks, silicon ratio and the quality metrics of both spectra.
    """
    ja = judge_bands(axis_a, y_a, bands)
    jb = judge_bands(axis_b, np.asarray(y_b, float) * scale_b, bands)
    qa, qb = spectrum_qc(axis_a, y_a), spectrum_qc(axis_b, np.asarray(y_b, float) * scale_b)
    rows = []
    for a, b in zip(ja, jb):
        row = {"center_cm1": a["center_cm1"], "role": a.get("role", "target"),
               "before_detected": a.get("detected"), "after_detected": b.get("detected")}
        ha, hb = a.get("net_height"), b.get("net_height")
        if ha is not None and hb is not None:
            na, nb = a.get("noise") or 0.0, b.get("noise") or 0.0
            row["change_sigma"] = round((hb - ha) / max(np.hypot(na, nb), 1e-9), 2)
            if a.get("detected"):
                row["ratio"] = round(hb / ha, 3) if ha else None
        rows.append(row)
    xa_a, _, snr_a = _residual_snr(axis_a, y_a)
    new = []
    for p in residual_peaks(axis_b, np.asarray(y_b, float) * scale_b):
        near = np.abs(xa_a - p["at_cm1"]) <= 10
        before = float(np.max(snr_a[near])) if near.any() else 0.0
        if before < 4.0 and p["snr"] >= 3 * max(before, 1.0):
            new.append({**p, "before_snr": round(before, 1)})
    si_ratio = (round(qb["si_height"] / qa["si_height"], 3)
                if qa.get("si_height") and qb.get("si_height") else None)
    return {"bands": rows, "new_peaks": new, "si_ratio": si_ratio,
            "before_qc": qa, "after_qc": qb, "judge_version": JUDGE_VERSION}
