"""Raman shift axis and material spectra of the virtual sample, built from the bands in config.json."""

from __future__ import annotations

import numpy as np

from virtual_benchmark.config import FWHM_MIN as _FWHM_MIN
from virtual_benchmark.config import TARGET_PEAK_CM1 as _TARGET_PEAK_CM1
from virtual_benchmark.config import BANDS, SCALE  # noqa: F401  (BANDS re-exported: session_log)


FWHM_MIN = float(_FWHM_MIN)

TARGET_PEAK_CM1 = float(_TARGET_PEAK_CM1)


GRATING_GROOVE = 1200.08440194926  # grooves/mm, Config.ini [TYPE-0] Groove
GRATING_CENTER_CM1 = 520.0  # cm-1, grating center position
LASER_NM = 532.021  # Config.ini RayleighWaveLength1
SI_PEAK_OFFSET = 0.00999999999999091  # Config.ini SiPeakOffset1
N_BINS = 1024  # Config.ini PixelCount


def calibrator():
    """Return the RamanCalibrator of this configuration (also attached to the virtual CCD)."""
    from backend.repo.hw_access.raman_calibration import RamanCalibrator
    return RamanCalibrator.from_factory_calibration(
        raman_center_cm1=GRATING_CENTER_CM1, laser_nm=LASER_NM,
        groove=GRATING_GROOVE, pixel_count=N_BINS, si_peak_offset=SI_PEAK_OFFSET)


_AXIS = np.asarray(calibrator().pixel_to_raman_shift(np.arange(N_BINS)), dtype=np.float64)

_AXIS.flags.writeable = False

AXIS_MIN, AXIS_MAX = float(_AXIS.min()), float(_AXIS.max())


def axis() -> np.ndarray:
    """Return the instrument's Raman shift axis (cm-1, 1024 points, read only)."""
    return _AXIS


KEYS = ("bg", "tgt2", "tgt")


def _lorentz(x: np.ndarray, center: float, height: float, fwhm: float) -> np.ndarray:
    """One Lorentzian line."""
    g = fwhm / 2.0
    return height * g ** 2 / ((x - center) ** 2 + g ** 2)


class SignalTable:
    """Material table that maps composition fractions to spectra."""

    def __init__(self, bands: dict | None = None, scale: dict | None = None,
                 fwhm_min: float | None = None, axis: np.ndarray | None = None):
        """Build the component spectra on the given axis (default: instrument axis)."""
        self.bands = BANDS if bands is None else bands
        self.scale = SCALE if scale is None else scale
        self.fwhm_min = FWHM_MIN if fwhm_min is None else float(fwhm_min)

        ax = _AXIS if axis is None else np.asarray(axis, dtype=np.float64)
        self.axis = ax
        rows = []
        for key in KEYS:
            spec = np.zeros(ax.shape, dtype=np.float64)
            for center, height, fwhm in self.bands[key]:
                spec += _lorentz(ax, center, height, max(fwhm, self.fwhm_min))
            rows.append(spec * float(self.scale[key]))

        self.M = np.stack(rows, axis=0)

    def mix(self, f) -> np.ndarray:
        """Return the spectrum for composition fractions f[..., 3] in counts/(%·s)."""
        f = np.asarray(f, dtype=np.float64)
        if f.shape[-1] != len(KEYS):
            raise ValueError(f"the last axis must have length {len(KEYS)} (= {KEYS}): {f.shape}")
        return f @ self.M

    def pure(self, key: str) -> np.ndarray:
        """Return the spectrum of a single material."""
        return self.M[KEYS.index(key)].copy()

    def __repr__(self) -> str:
        n = {k: len(self.bands[k]) for k in KEYS}
        return f"<SignalTable bands={n} scale={self.scale} fwhm_min={self.fwhm_min}>"


def _check_config() -> None:
    """Check at import time that the configuration matches the band table."""
    centers = [c for c, _, _ in BANDS["tgt"]]
    if not any(abs(c - TARGET_PEAK_CM1) < 1e-6 for c in centers):
        raise ValueError(
            f"target_peak_cm1={TARGET_PEAK_CM1} in config.json is not a tgt band: {centers}")
    if not (AXIS_MIN < TARGET_PEAK_CM1 < AXIS_MAX):
        raise ValueError(f"target_peak_cm1={TARGET_PEAK_CM1} is outside the axis ({AXIS_MIN}~{AXIS_MAX})")
    if set(BANDS) != set(KEYS) or set(SCALE) != set(KEYS):
        raise ValueError(f"BANDS/SCALE keys differ from KEYS ({KEYS})")


_check_config()
