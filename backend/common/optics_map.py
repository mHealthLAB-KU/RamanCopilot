"""Conversion between camera pixel coordinates and stage coordinates."""

from __future__ import annotations

from backend.common.config.hw import CAMERA_WIDTH, CAMERA_HEIGHT, DEFAULT_OBJECTIVE, LENSES, Lens

SIGN_X = -1  # pixel +X moves the stage -X
SIGN_Y = +1  # pixel +Y moves the stage +Y

_objective: str = DEFAULT_OBJECTIVE


def objective() -> str:
    """Return the name of the current objective ("20x", "50x", "100x")."""
    return _objective


def lens() -> Lens:
    """Return the field of view and calibration factors of the current objective."""
    return LENSES[_objective]


def set_objective(name: str) -> None:
    """Switch the objective and reset the laser spot to the image center."""
    global _objective
    key = str(name).strip().lower()
    if key not in LENSES:
        raise ValueError(f"Unknown objective {name!r}. Available: {list(LENSES)}")
    _objective = key
    reset_aim_px()


_aim_view_px: tuple[float, float] | None = None


def set_aim_px(x: float, y: float) -> None:
    """Set the laser spot position in view-resolution pixels."""
    global _aim_view_px
    _aim_view_px = (float(x), float(y))


def reset_aim_px() -> None:
    """Reset the laser spot to the image center."""
    global _aim_view_px
    _aim_view_px = None


def aim_px(width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """Return the laser spot pixel, optionally scaled to a given image resolution."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    if _aim_view_px is None:
        return (w / 2.0, h / 2.0)
    return (_aim_view_px[0] * w / CAMERA_WIDTH, _aim_view_px[1] * h / CAMERA_HEIGHT)


def mm_per_px(width: int | None = None, height: int | None = None,
              objective: str | None = None) -> tuple[float, float]:
    """Return the stage distance (mm, positive) covered by one image pixel (current objective by default)."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    ln = lens() if objective is None else LENSES[objective]
    return (ln.width_um  / w * ln.calib_x / 1000.0,
            ln.height_um / h * ln.calib_y / 1000.0)


def fov_mm(width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """Return the physical size (mm) of the camera field of view."""
    mx, my = mm_per_px(width, height)
    return (mx * int(width or CAMERA_WIDTH), my * int(height or CAMERA_HEIGHT))


def pixel_delta_to_mm(dpx: float, dpy: float,
                      width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """Convert a pixel displacement to a signed stage displacement in mm."""
    mx, my = mm_per_px(width, height)
    return (dpx * mx * SIGN_X, dpy * my * SIGN_Y)


def pixel_to_stage(px: float, py: float, cur_x: float, cur_y: float,
                   width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """Convert a pixel coordinate to the absolute stage coordinate (mm) that brings it under the laser spot."""
    ax, ay = aim_px(width, height)
    dx_mm, dy_mm = pixel_delta_to_mm(px - ax, py - ay, width, height)
    return (cur_x + dx_mm, cur_y + dy_mm)


def stage_to_pixel(sx: float, sy: float, cur_x: float, cur_y: float,
                   width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """Project a stage coordinate onto the current image (inverse of pixel_to_stage)."""
    ax, ay = aim_px(width, height)
    mx, my = mm_per_px(width, height)
    return (ax + (sx - cur_x) * SIGN_X / mx,
            ay + (sy - cur_y) * SIGN_Y / my)


def scene_extent(cur_x: float, cur_y: float,
                 width: int | None = None, height: int | None = None) -> list[float]:
    """Return the stage range [xmin, xmax, ymin, ymax] (mm) covered by the current image."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    x0, y0 = pixel_to_stage(0, 0, cur_x, cur_y, w, h)
    x1, y1 = pixel_to_stage(w, h, cur_x, cur_y, w, h)
    return [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
