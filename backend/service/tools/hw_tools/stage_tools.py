"""Agent tools for the XYZ stage."""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field

from backend.common.config.hw import STAGE_MAX_X, STAGE_MAX_Y, STAGE_MAX_Z, STAGE_MIN_Z
from backend.repo.hw_access.hao.USE_stage_test import (
    MAX_SPEED_XY as _MAX_SPEED_XY,
    MAX_SPEED_Z as _MAX_SPEED_Z,
    MIN_SPEED_MM_S as _MIN_SPEED_MM_S,
)
from backend.repo.hw_access import hw_core as _hw
from backend.repo.hw_access.hw_core import (
    _STAGE_BUSY,
    _stage_read,
)
from backend.service.tools.hw_tools.guards import (
    _check_stage_target,
    _serialized,
    _stage_unavailable,
)
from backend.common.result import fail, ok


def get_stage_speed() -> dict:
    """Return the stage speed of each axis.

    Returns:
        dict: ok() with x, y and z speed in mm/s, or fail().
    """
    err = _stage_unavailable()
    if err:
        return err

    try:
        vel = _stage_read(lambda: _hw._stage.get_velocity())

        if vel is _STAGE_BUSY:
            return fail("The stage is being connected or released right now, so its speed was not read. "
                        "Nothing was changed - try again in a few seconds.")

        if not (isinstance(vel, dict) and vel.get("ok")):
            return fail(vel.get("error", "Failed to read velocity") if isinstance(vel, dict)
                        else "Unexpected velocity type")

        return ok(x_speed_mm_s=vel["x_speed_mm_s"],
                  y_speed_mm_s=vel["y_speed_mm_s"],
                  z_speed_mm_s=vel["z_speed_mm_s"])
    except Exception as e:
        return fail(str(e))


def get_stage_position() -> dict:
    """Return the current stage position.

    Returns:
        dict: ok() with x, y and z in mm, or fail().
    """
    err = _stage_unavailable()
    if err:
        return err

    try:
        pos = _stage_read(lambda: _hw._stage.get_position())

        if pos is _STAGE_BUSY:
            return fail("The stage is being connected or released right now, so its position was not read. "
                        "Nothing was changed - try again in a few seconds.")
        if pos is None:
            return fail("Failed to query stage position")

        return ok(x=pos[0], y=pos[1], z=pos[2])
    except Exception as e:
        return fail(str(e))


@_serialized("set_stage_speed")
def set_stage_speed(
    x_speed_mm_s: Annotated[Optional[float], Field(description='X-axis movement speed (mm/s, max 5.0). Optional.')] = None,
    y_speed_mm_s: Annotated[Optional[float], Field(description='Y-axis movement speed (mm/s, max 5.0). Optional.')] = None,
    z_speed_mm_s: Annotated[Optional[float], Field(description='Z-axis movement speed (mm/s, max 0.1). Optional.')] = None,
) -> dict:
    """Set the stage speed of each axis. Omitted axes keep their current speed.

    Returns:
        dict: ok() with the speeds that will be applied, or fail().
    """
    err = _stage_unavailable()
    if err:
        return err

    _bad = {n: v for n, v in (("x_speed_mm_s", x_speed_mm_s),
                              ("y_speed_mm_s", y_speed_mm_s),
                              ("z_speed_mm_s", z_speed_mm_s))
            if v is not None and abs(float(v)) < _MIN_SPEED_MM_S}
    if _bad:
        return fail(
            f"Rejected: {', '.join(f'{n}={v}' for n, v in _bad.items())}. "
            f"A speed of zero (or near zero) is NOT 'hold this axis still' - it means the "
            f"axis can never reach a target, so the next move that needs this axis would "
            f"never finish and the stage would have to be power-cycled. Nothing was changed. "
            f"To leave an axis alone, OMIT its parameter instead of passing 0. "
            f"Valid range: {_MIN_SPEED_MM_S} to {_MAX_SPEED_XY} mm/s for X and Y, "
            f"{_MIN_SPEED_MM_S} to {_MAX_SPEED_Z} mm/s for Z.",
            rejected=_bad, min_speed_mm_s=_MIN_SPEED_MM_S)

    try:
        current_vel = _hw._stage.get_velocity()
        if not current_vel.get("ok"):
            return fail(current_vel.get("error", "Failed to read current velocity"))

        req = {
            "x_speed_mm_s": x_speed_mm_s if x_speed_mm_s is not None else current_vel["x_speed_mm_s"],
            "y_speed_mm_s": y_speed_mm_s if y_speed_mm_s is not None else current_vel["y_speed_mm_s"],
            "z_speed_mm_s": z_speed_mm_s if z_speed_mm_s is not None else current_vel["z_speed_mm_s"],
        }

        limits = {"x_speed_mm_s": _MAX_SPEED_XY,
                  "y_speed_mm_s": _MAX_SPEED_XY,
                  "z_speed_mm_s": _MAX_SPEED_Z}
        eff, clipped = {}, {}
        for k, v in req.items():
            hi = limits[k]
            e = max(-hi, min(hi, float(v)))
            eff[k] = e
            if e != float(v):
                clipped[k] = {"requested": float(v), "applied": e, "limit_mm_s": hi}

        applied = _hw._stage.set_velocity(eff["x_speed_mm_s"], eff["y_speed_mm_s"],
                                          eff["z_speed_mm_s"], 0.0)
        if applied is False:
            return fail("The controller rejected the velocity command, so the stage speed was NOT changed. "
                        "Check the stage connection and retry.")

        out = ok(**eff)
        if clipped:
            out["clipped"] = clipped
            out["note"] = ("Some axes were clipped to the hardware speed limit "
                           f"(XY <= {_MAX_SPEED_XY} mm/s, Z <= {_MAX_SPEED_Z} mm/s). "
                           "The values reported here are what the stage will actually use.")
        return out

    except Exception as e:
        return fail(str(e))


@_serialized("move_stage")
def move_stage(
    x: Annotated[float, Field(ge=0, le=STAGE_MAX_X, description=f'X-axis position (mm, 0-{STAGE_MAX_X})')],
    y: Annotated[float, Field(ge=0, le=STAGE_MAX_Y, description=f'Y-axis position (mm, 0-{STAGE_MAX_Y})')],
    z: Annotated[Optional[float], Field(ge=STAGE_MIN_Z, le=STAGE_MAX_Z, description=f'Z-axis position (mm, {STAGE_MIN_Z}-{STAGE_MAX_Z}). Optional - omit to keep the current Z.')] = None,
) -> dict:
    """Move the stage to absolute coordinates. Targets out of range are refused.

    Returns:
        dict: ok() with the position read back after the move, or fail().
    """
    err = _stage_unavailable()
    if err:
        return err

    err = _check_stage_target(x, y, z)
    if err:
        return err

    try:
        kw = {"x": x, "y": y, "wait": True}
        if z is not None:
            kw["z"] = z
        else:
            kw["z"] = _hw._stage.get_position()[2]

        moved = _hw._stage.move_absolute(**kw)

        pos = _hw._stage.get_position()
        if pos is None:
            return fail("The stage did not report a position after the move.")
        if not moved:
            return fail(
                f"The stage driver rejected the move to (x={x}, y={y}, z={kw['z']}) and "
                f"reported failure. The stage is NOT at the requested position - it is at "
                f"the position reported below, which may be where it started or somewhere "
                f"in between. This is a hardware/driver fault, not a problem with your "
                f"arguments: do not retry with different coordinates expecting a different "
                f"result. Report the failure and the actual position.",
                position={"x": pos[0], "y": pos[1], "z": pos[2]})
        return ok(position={"x": pos[0], "y": pos[1], "z": pos[2]})
    except Exception as e:
        return fail(str(e))


@_serialized("move_stage_relative")
def move_stage_relative(
    dx: Annotated[Optional[float], Field(description='Displacement in X (mm)')] = 0.0,
    dy: Annotated[Optional[float], Field(description='Displacement in Y (mm)')] = 0.0,
    dz: Annotated[Optional[float], Field(description='Displacement in Z (mm)')] = 0.0,
) -> dict:
    """Move the stage by a relative displacement.

    Returns:
        dict: ok() with the position after the move, or fail().
    """
    err = _stage_unavailable()
    if err:
        return err

    try:
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")

        err = _check_stage_target(float(pos[0]) + float(dx),
                                  float(pos[1]) + float(dy),
                                  float(pos[2]) + float(dz))
        if err:
            err["error"] += (f" - this is the target after applying the relative move "
                             f"(dx={dx}, dy={dy}, dz={dz}) to the current position.")
            return err

        moved = _hw._stage.move_relative(dx, dy, dz, 0)

        pos = _hw._stage.get_position()
        if pos is None:
            return fail("The stage did not report a position after the move.")
        if not moved:
            return fail(
                f"The stage driver rejected the relative move (dx={dx}, dy={dy}, dz={dz}) "
                f"and reported failure. The stage did NOT complete the requested "
                f"displacement - it is at the position reported below, which may be where "
                f"it started or somewhere in between. This is a hardware/driver fault, not "
                f"a problem with your arguments: do not retry with a different displacement "
                f"expecting a different result. Report the failure and the actual position.",
                position={"x": pos[0], "y": pos[1], "z": pos[2]})
        return ok(position={"x": pos[0], "y": pos[1], "z": pos[2]})
    except Exception as e:
        return fail(str(e))
