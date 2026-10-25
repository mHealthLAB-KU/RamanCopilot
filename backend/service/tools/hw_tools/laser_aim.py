"""Measures where the guide-beam spot falls on the camera image."""

from __future__ import annotations

import math

from backend.common import optics_map as _om
from backend.common import vision as _vis
from backend.common.config.hw import CAMERA_HEIGHT, CAMERA_WIDTH
from backend.repo.hw_access import hw_core as _hw
from backend.repo.store import aim_store as _store

AIM_MAX_OFFSET_PX = 80.0  # spots farther from the image center are rejected


def _describe(px: tuple[float, float]) -> dict:
    """Describe one laser spot position, including its offset from the image center."""
    cx, cy = CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0
    return {"pixel": [round(px[0], 1), round(px[1], 1)],
            "offset_from_center_px": [round(px[0] - cx, 1), round(px[1] - cy, 1)]}


def locate_laser_aim() -> dict:
    """Measure the guide-beam spot center at the current Z and apply it as the laser spot."""
    kept = _describe(_om.aim_px())

    diff = _vis.capture_laser_diff(_hw._camera, _hw._laser, n_avg=3)
    if not diff.get("ok") or diff.get("spot_px") is None:
        return {"updated": False, **kept,
                "reason": "The guide-beam spot could not be located, so the aim point is unchanged."}

    spot = _vis.sensor_to_view_px(*diff["spot_px"], *diff["frame_size"])
    off = math.hypot(spot[0] - CAMERA_WIDTH / 2.0, spot[1] - CAMERA_HEIGHT / 2.0)
    if off > AIM_MAX_OFFSET_PX:
        return {"updated": False, **kept,
                "reason": (f"The brightest spot was found {off:.0f}px from the image centre "
                           f"(limit {AIM_MAX_OFFSET_PX:.0f}px), which is too far to be the guide "
                           f"beam - likely a reflection. The aim point is unchanged.")}

    _om.set_aim_px(*spot)
    pos = _hw._stage.get_position() if _hw._stage is not None else None
    _store.save(_om.objective(), *spot,
                z=round(float(pos[2]), 4) if pos else None,
                area_px=int(diff["area_px"]))
    return {"updated": True, **_describe(spot), "spot_area_px": int(diff["area_px"])}
