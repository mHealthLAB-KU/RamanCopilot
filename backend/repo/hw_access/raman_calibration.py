"""Pixel to wavelength and Raman shift calibration of the spectrometer."""
from __future__ import annotations

import configparser
import math
from pathlib import Path

import numpy as np

__all__ = ["RamanCalibrator", "wl_p_calib"]

def _default_config_path() -> Path:
    """Return the Config.ini path from backend.common.config.hw (resolved at call time)."""
    from backend.common.config.hw import CONFIG_PATH
    return Path(CONFIG_PATH)


def wl_p_calib(px, n0, offset_adjust, wl_center, m_order, d_grating,
               x_pixel, f, delta, gamma, curvature=0.0):
    """Convert pixel indices to wavelength with the Czerny-Turner grating equation."""
    n = px - (n0 + offset_adjust * wl_center)
    psi = np.arcsin(m_order * wl_center / (2 * d_grating * np.cos(gamma / 2)))
    eta = np.arctan(n * x_pixel * np.cos(delta) / (f + n * x_pixel * np.sin(delta)))
    return (
        (d_grating / m_order)
        * (np.sin(psi - 0.5 * gamma) + np.sin(psi + 0.5 * gamma + eta))
        + curvature * n ** 2
    )


class RamanCalibrator:
    """Converts CCD pixel indices to wavelength and Raman shift."""
    def __init__(self, laser_nm: float, lut_cm1: np.ndarray):
        self.laser_nm = float(laser_nm)
        self._lut = lut_cm1
        self._wl_nm: np.ndarray | None = None

    def pixel_to_wavelength(self, px) -> np.ndarray:
        return self._wl_nm[np.asarray(px, dtype=int)]

    def pixel_to_raman_shift(self, px) -> np.ndarray:
        return self._lut[np.asarray(px, dtype=int)]

    @classmethod
    def from_factory_calibration(cls, config_path=None,
                                  raman_center_cm1: float | None = None,
                                  laser_nm: float | None = None,
                                  f_mm: float | None = None,
                                  pixel_count: int | None = None,
                                  pixel_width_um: float | None = None,
                                  groove: float | None = None,
                                  tilt_angle_deg: float | None = None,
                                  si_peak_offset: float | None = None,
                                  pixel_offset: float = 0.0) -> "RamanCalibrator":
        """Build a calibrator from the factory calibration in Config.ini, with an optional pixel offset."""
        path = Path(config_path) if config_path else _default_config_path()

        cp = configparser.ConfigParser(strict=False)
        with open(path, encoding="cp949", errors="ignore") as fh:
            cp.read_file(fh)

        auto = cp["AUTO_CALIBRATION"]
        selected_type = int(auto.get("SelectedType", "0"))
        selected_rayleigh = int(auto.get("SelectedRayleigh", "0"))

        if pixel_count is None:
            pixel_count = int(auto.get("PixelCount", "1024"))
        if pixel_width_um is None:
            pixel_width_um = float(auto.get("PixelWidth", "23.8902"))
        x_pixel_mm = pixel_width_um / 1000.0

        if laser_nm is None:
            rayleigh_key = f"RayleighWaveLength{selected_rayleigh + 1}"
            laser_nm = float(auto.get(rayleigh_key, "532.021"))

        type_section = f"TYPE-{selected_type}"
        t = cp[type_section]

        if groove is None:
            groove = float(t["Groove"])
        d_grating_mm = 1.0 / groove
        n0 = pixel_count // 2 + pixel_offset
        gamma = math.radians(float(t.get("dV", "14.755")))
        if tilt_angle_deg is None:
            tilt_angle_deg = float(t.get("TiltAngle", "0"))
        delta = math.radians(tilt_angle_deg)
        if f_mm is None:
            f_mm = float(t["FocalLength"])
        if si_peak_offset is None:
            si_peak_offset = 0.0

        grating_center_nm = 1.0 / (1.0 / laser_nm - raman_center_cm1 / 1e7)
        wl_center_mm = grating_center_nm * 1e-6

        pixels = np.arange(pixel_count, dtype=float)
        wl_mm = wl_p_calib(
            pixels,
            n0=n0,
            offset_adjust=si_peak_offset,
            wl_center=wl_center_mm,
            m_order=1,
            d_grating=d_grating_mm,
            x_pixel=x_pixel_mm,
            f=f_mm,
            delta=delta,
            gamma=gamma,
            curvature=0.0,
        )
        wl_nm = wl_mm * 1e6
        lut_cm1 = (1.0 / laser_nm - 1.0 / wl_nm) * 1e7

        cal = cls(laser_nm=laser_nm, lut_cm1=lut_cm1)
        cal._wl_nm = wl_nm
        return cal
