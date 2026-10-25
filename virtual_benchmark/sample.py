"""Virtual sample that returns a spectrum and damage at each stage position."""

from __future__ import annotations

import numpy as np

from backend.common.optics_map import mm_per_px as _mm_per_px
from virtual_benchmark.bands import KEYS, SignalTable, axis
from virtual_benchmark.config import (  # noqa: F401  (FULL_WELL, READ_SIGMA re-exported: rig)
    ALPHA, BEST_EXPOSURE_S, BEST_POWER_PCT, COLORS_RGB, FULL_WELL, PEAK_MAX_COUNTS,
    READ_SIGMA, REF_OBJECTIVE, SPOT_W_UM, TARGET_AREA, W_DECAY,
)
from virtual_benchmark.world import FIELD_DS, MAP_H, MAP_W, VirtualWorld


MAT_RGB = np.array([COLORS_RGB[k] for k in KEYS]) / 255.0


DAMAGE_W_UM = 2.0 * SPOT_W_UM  # um, damage spreads wider than the beam because of heat

VIEW_SUPERSAMPLE = 4  # samples per camera pixel along each axis

_MM_PER_PX_X, _MM_PER_PX_Y = _mm_per_px(objective=REF_OBJECTIVE)
_UM_PER_PX_X, _UM_PER_PX_Y = _MM_PER_PX_X * 1000.0, _MM_PER_PX_Y * 1000.0

_CUT_PX_X = 2.5 * DAMAGE_W_UM / _UM_PER_PX_X
_CUT_PX_Y = 2.5 * DAMAGE_W_UM / _UM_PER_PX_Y


class VirtualSample:
    """Seeded virtual sample queried by stage coordinates (mm)."""

    def __init__(self, seed: int, target_area: float = TARGET_AREA,
                 table: SignalTable | None = None):
        self.world = VirtualWorld(seed, target_area)
        self.table = _REF_TABLE if table is None else table

        self.shots: list[tuple[float, float, float]] = []


    def composition_at(self, x_mm, y_mm) -> np.ndarray:
        """Return (f_bg, f_tgt2, f_tgt) at stage coordinates."""
        return self.world.composition_at(x_mm, y_mm)

    def spectrum_at(self, x_mm, y_mm, axis_cm1=None) -> np.ndarray:
        """Return the spectrum at a position in counts/(%·s)."""
        return self.spectrum_rate(self.composition_at(x_mm, y_mm), axis_cm1)

    def spectrum_rate(self, f, axis_cm1=None) -> np.ndarray:
        """Convert composition fractions to a spectrum in counts/(%·s)."""
        if axis_cm1 is None:
            return self.table.mix(f)
        return SignalTable(bands=self.table.bands, scale=self.table.scale,
                           fwhm_min=self.table.fwhm_min, axis=axis_cm1).mix(f)

    def band_map(self, cm1: float, w: int, h: int) -> np.ndarray:
        """Return the intensity map (h, w) at one wavenumber."""
        k = int(np.argmin(np.abs(self.table.axis - float(cm1))))
        return self.world._sample_field_grid(w, h) @ self.table.M[:, k]


    def _view_rgb(self, x_mm, y_mm, w: int, h: int) -> np.ndarray:
        """Render the composition covered by w×h pixels as RGB at the current objective (outside the map is black)."""
        col_c, row_c = self.world.stage_to_map(x_mm, y_mm)
        mx, my = _mm_per_px(w, h)
        zx, zy = mx / self.world.mm_x, my / self.world.mm_y
        cols = col_c - w / 2.0 * zx + np.arange(w) * zx
        rows = row_c - h / 2.0 * zy + np.arange(h) * zy
        cc, rr = np.meshgrid(cols, rows)
        f = burn(self.world._sample_field(cc, rr), self._survival_grid(cc, rr))
        rgb = np.clip(f @ MAT_RGB, 0.0, 1.0)
        inside = (cc >= 0) & (cc < MAP_W) & (rr >= 0) & (rr < MAP_H)
        return rgb * inside[..., None]

    def _survival_grid(self, cols, rows, supersample: int = VIEW_SUPERSAMPLE) -> np.ndarray:
        """Return the survival fraction on the map pixel grid."""
        if supersample <= 1:
            return survival(self.dose_map(cols, rows))
        off = (np.arange(supersample) + 0.5) / supersample - 0.5
        acc = np.zeros(np.broadcast_shapes(np.shape(cols), np.shape(rows)), dtype=np.float64)
        for dr in off:
            for dc in off:
                acc += survival(self.dose_map(np.asarray(cols) + dc, np.asarray(rows) + dr))
        return acc / (supersample * supersample)

    def crop(self, cur_x: float, cur_y: float, out_w: int, out_h: int) -> np.ndarray:
        """Return the field of view at the current stage position as BGR uint8."""
        rgb = self._view_rgb(cur_x, cur_y, int(out_w), int(out_h))
        return (rgb * 255.0).astype(np.uint8)[..., ::-1].copy()

    @property
    def center_mm(self) -> tuple[float, float]:
        """Return the sample center."""
        return (float(self.world.center_x), float(self.world.center_y))

    def color_at(self, x_mm: float, y_mm: float) -> tuple[int, int, int]:
        """Return the (B, G, R) color at a position."""
        b = self.bounds_mm()
        if not (b[0] <= x_mm <= b[1] and b[2] <= y_mm <= b[3]):
            return (0, 0, 0)
        rgb = np.clip(self.composition_at(x_mm, y_mm) @ MAT_RGB, 0.0, 1.0) * 255.0
        r, g, bl = (int(round(v)) for v in rgb)
        return (bl, g, r)


    def dose_map(self, cols, rows) -> np.ndarray:
        """Return the accumulated dose on map pixel coordinates (Gaussian beam profile)."""
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        out = np.zeros(np.broadcast_shapes(cols.shape, rows.shape), dtype=np.float64)
        for sc, sr, peak in self.shots:
            near = ((np.abs(cols - sc) <= _CUT_PX_X) & (np.abs(rows - sr) <= _CUT_PX_Y))
            if not near.any():
                continue
            dx = (np.broadcast_to(cols, out.shape)[near] - sc) * _UM_PER_PX_X
            dy = (np.broadcast_to(rows, out.shape)[near] - sr) * _UM_PER_PX_Y
            out[near] += peak * np.exp(-2.0 * (dx * dx + dy * dy) / (DAMAGE_W_UM ** 2))
        return out

    def dose_at(self, x_mm, y_mm) -> float:
        """Return the dose accumulated at a position so far."""
        col, row = self.world.stage_to_map(x_mm, y_mm)
        return float(self.dose_map(col, row))

    def survival_at(self, x_mm, y_mm) -> float:
        """Return the current survival fraction at a position (1.0 = undamaged)."""
        return survival(self.dose_at(x_mm, y_mm))

    def burned_spots(self) -> int:
        """Return the number of burned spots."""
        return int(sum(survival(peak) < 0.5 for _, _, peak in self.shots))

    def measure(self, x_mm, y_mm, power_pct: float, exposure_s: float, *,
                noise: bool = True, axis_cm1=None, rng=None) -> np.ndarray:
        """Fire the laser and return a spectrum; the dose accumulates irreversibly."""
        before = self.dose_at(x_mm, y_mm)
        out = spectrum(self, x_mm, y_mm, power_pct, exposure_s,
                       noise=noise, axis_cm1=axis_cm1, dose_before=before, rng=rng)
        col, row = self.world.stage_to_map(x_mm, y_mm)
        self.shots.append((float(col), float(row), dose_of(power_pct, exposure_s)))
        return out


    def bounds_mm(self):
        """Return the stage range [xmin, xmax, ymin, ymax] (mm) covered by the sample."""
        return self.world.bounds_mm()

    def extreme_xy(self, key: str = "tgt", how: str = "max"):
        """Return the stage coordinates where a component is most/least concentrated."""
        comp = self.world.field[..., KEYS.index(key)]
        fr, fc = np.unravel_index(np.argmax(comp) if how == "max" else np.argmin(comp),
                                  comp.shape)
        x, y = self.world.map_to_stage((fc + 0.5) * FIELD_DS, (fr + 0.5) * FIELD_DS)
        return float(x), float(y)

    def __repr__(self) -> str:
        b = self.bounds_mm()
        return (f"<VirtualSample seed={self.world.seed} "
                f"x {b[0]:.2f}~{b[1]:.2f} y {b[2]:.2f}~{b[3]:.2f} mm>")


E_TH = BEST_EXPOSURE_S

_REF_TABLE = SignalTable()
_BRIGHTEST_RATE = float(_REF_TABLE.M.max())

G = PEAK_MAX_COUNTS / (BEST_POWER_PCT * BEST_EXPOSURE_S * _BRIGHTEST_RATE)


def dose_of(power_pct: float, exposure_s: float) -> float:
    """Return the dose of one irradiation (superlinear in power)."""
    return float((power_pct / BEST_POWER_PCT) ** ALPHA * exposure_s)


def survival(dose):
    """Return the surviving fraction for a dose (1.0 = undamaged)."""
    u = np.asarray(dose, dtype=np.float64) / E_TH
    A = np.exp(-np.maximum(0.0, u - 1.0) / W_DECAY)
    return float(A) if A.ndim == 0 else A


def burn(f, A: float) -> np.ndarray:
    """Apply survival to the composition, transferring burned material to the substrate."""
    f = np.asarray(f, dtype=np.float64)
    f_bg, f_tgt2, f_tgt = f[..., 0], f[..., 1], f[..., 2]
    moved = (f_tgt2 + f_tgt) * (1.0 - A)
    return np.stack([f_bg + moved, f_tgt2 * A, f_tgt * A], axis=-1)


def spectrum(sample: "VirtualSample", x_mm, y_mm, power_pct: float, exposure_s: float, *,
             noise: bool = True, axis_cm1=None, dose_before: float = 0.0,
             rng=None) -> np.ndarray:
    """Return the spectrum in counts for an irradiation at a position."""
    A = survival(dose_before + dose_of(power_pct, exposure_s))
    f_eff = burn(sample.composition_at(x_mm, y_mm), A)

    rate = sample.spectrum_rate(f_eff, axis_cm1)
    counts = G * float(power_pct) * float(exposure_s) * rate

    if noise:
        rng = np.random.default_rng() if rng is None else rng
        counts = (counts
                  + rng.normal(0.0, np.sqrt(np.maximum(counts, 1.0)))
                  + rng.normal(0.0, READ_SIGMA, counts.shape))
    return np.clip(counts, 0.0, FULL_WELL)
