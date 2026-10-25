"""Composition field of the virtual sample."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, zoom

from backend.common.optics_map import mm_per_px as _mm_per_px
from backend.common.config.hw import (
    CAMERA_HEIGHT, CAMERA_WIDTH, STAGE_CENTER_X, STAGE_CENTER_Y,
)
from virtual_benchmark.config import FBM_BASE, FBM_OCTAVES, GAMMA, MAP_H, MAP_W, REF_OBJECTIVE, TARGET_AREA


FIELD_DS = 4  # downsampling of the stored composition field


def _fbm(rng, w, h, octaves=FBM_OCTAVES, base=FBM_BASE) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gw = max(2, int(base * 2 ** o))
        gh = max(2, int(round(gw * h / w)))
        g = rng.random((gh, gw))
        up = zoom(g, (h / gh, w / gw), order=(3 if o < 2 else 1), mode="nearest")
        if up.shape[0] < h or up.shape[1] < w:
            up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")
        out += amp * up[:h, :w]
        total += amp
        amp *= 0.5
    return _rank_uniform(out / total)


def _rank_uniform(a: np.ndarray) -> np.ndarray:
    """Rank-transform values to a uniform distribution on [0, 1]."""
    flat = a.ravel().astype(np.float32, copy=False)
    order = np.argsort(flat)
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.arange(flat.size, dtype=np.float32)
    return (ranks / max(flat.size - 1, 1)).reshape(a.shape)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class VirtualWorld:
    """Seeded composition field of one virtual sample."""

    def __init__(self, seed: int, target_area: float = TARGET_AREA):
        """Create the field; target_area is the exact fraction of the map with f_tgt > 0.5."""
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        self.mm_x, self.mm_y = _mm_per_px(CAMERA_WIDTH, CAMERA_HEIGHT, objective=REF_OBJECTIVE)
        self.center_x, self.center_y = float(STAGE_CENTER_X), float(STAGE_CENTER_Y)

        fw, fh = MAP_W // FIELD_DS, MAP_H // FIELD_DS
        n1 = _fbm(rng, fw, fh)
        n2 = _fbm(rng, fw, fh, base=max(2, FBM_BASE - 2))

        span = 1.0 - 0.5 ** (1.0 / GAMMA)
        theta = float(np.clip(1.0 - target_area / span, 0.0, 0.95))
        self.target_area, self.theta = float(target_area), theta

        f_tgt = np.clip((n1 - theta) / (1.0 - theta), 0.0, 1.0) ** GAMMA
        f_tgt2 = (1.0 - f_tgt) * _smoothstep(n2)
        f_bg = 1.0 - f_tgt - f_tgt2
        self.field = np.stack([f_bg, f_tgt2, f_tgt], axis=-1).astype(np.float32)


    def stage_to_map(self, x_mm, y_mm):
        """Convert stage mm to map pixels (X is flipped)."""
        col = MAP_W / 2.0 - (np.asarray(x_mm, dtype=np.float64) - self.center_x) / self.mm_x
        row = MAP_H / 2.0 + (np.asarray(y_mm, dtype=np.float64) - self.center_y) / self.mm_y
        return col, row

    def map_to_stage(self, col, row):
        """Convert map pixels to stage mm."""
        x = self.center_x - (np.asarray(col, dtype=np.float64) - MAP_W / 2.0) * self.mm_x
        y = self.center_y + (np.asarray(row, dtype=np.float64) - MAP_H / 2.0) * self.mm_y
        return x, y

    def bounds_mm(self):
        """Return the stage range [xmin, xmax, ymin, ymax] (mm) covered by the world."""
        x0, y0 = self.map_to_stage(0, 0)
        x1, y1 = self.map_to_stage(MAP_W, MAP_H)
        return [float(min(x0, x1)), float(max(x0, x1)),
                float(min(y0, y1)), float(max(y0, y1))]

    def composition_at(self, x_mm, y_mm) -> np.ndarray:
        """Return (f_bg, f_tgt2, f_tgt) at stage coordinates."""
        col, row = self.stage_to_map(x_mm, y_mm)
        return self._sample_field(col, row)


    def _sample_field(self, cols, rows):
        """Bilinearly sample the composition field at map pixel coordinates."""
        fc = np.asarray(cols, dtype=np.float64) / FIELD_DS
        fr = np.asarray(rows, dtype=np.float64) / FIELD_DS
        shape = np.broadcast_shapes(fc.shape, fr.shape)
        fc, fr = np.broadcast_to(fc, shape).ravel(), np.broadcast_to(fr, shape).ravel()
        out = [map_coordinates(self.field[..., k], [fr, fc], order=1, mode="nearest")
               for k in range(3)]
        return np.stack(out, axis=-1).reshape(*shape, 3)

    def _sample_field_grid(self, w, h):
        """Resample the field onto a w×h grid."""
        cols = (np.arange(w) + 0.5) * (MAP_W / w)
        rows = (np.arange(h) + 0.5) * (MAP_H / h)
        cc, rr = np.meshgrid(cols, rows)
        return self._sample_field(cc, rr)
