"""Agent tools for the laser."""

from __future__ import annotations

from backend.common.result import fail, ok
from pydantic import Field
from typing import Annotated
from backend.repo.hw_access import hw_core as _hw
from backend.service.tools.hw_tools.guards import _serialized
from backend.service.tools.hw_tools.hw_ops import POWER_RANGE_TXT, _apply_laser_power, _beam_state, _restore_guide_beam_quiet


@_serialized("laser_on")
def laser_on() -> dict:
    """Turn the laser on.

    The measurement beam is emitted only if a power has been applied. Otherwise the guide beam is emitted.

    Returns:
        dict: ok() with the emitted beam and power, or fail().
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    st = _beam_state()
    armed, beam = st["power_armed"], st["beam"]

    try:
        _hw._laser.laser_on()
    except Exception as e:
        return fail(str(e))

    out = ok(status=f"Laser ON ({beam} beam)", beam=beam)
    if armed:
        out["power_mw"] = st["power_mw"]
        out["power_pct"] = st["power_pct"]
    else:
        out["note"] = (
            "Only the GUIDE beam is emitted - the laser power has not been applied to the optics, "
            "so the measurement beam is still blocked by the ND filter. For a real measurement call "
            "set_laser_power(mw) first, or use acquire_spectrum(power=...) which does "
            "power -> on -> acquire -> off atomically.")
    return out


@_serialized("laser_off")
def laser_off() -> dict:
    """Stop laser emission. The power setting and optics position are kept.

    Returns:
        dict: ok() with the power state, or fail() if the stop was not confirmed.
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    try:
        confirmed = _hw._laser.laser_off()
        st = _beam_state()

        if confirmed is False:
            blocked = _restore_guide_beam_quiet()
            return fail("Laser stop (SSPW 0) was NOT confirmed - the beam may still be on. " +
                        ("The ND filter has been moved to the blocking position. " if blocked else
                         "ND blocking could not be confirmed either. ") +
                        "Check the laser controller link.",
                        laser_off_unconfirmed=True,
                        nd_blocked=bool(blocked),
                        power_armed=st["power_armed"])

        return ok(status="Laser OFF",
                  power_armed=st["power_armed"],
                  power_mw=st["power_mw"],
                  power_pct=st["power_pct"])
    except Exception as e:
        return fail(str(e))


@_serialized("set_laser_power")
def set_laser_power(
    mw: Annotated[float, Field(description=f'Laser power, {POWER_RANGE_TXT}.')],
) -> dict:
    """Set the laser power and arm the measurement beam without firing.

    Returns:
        dict: ok() with the beam state, or fail().
    """
    err = _apply_laser_power(mw)
    if err:
        return err
    return ok(**_beam_state())


def get_laser_status() -> dict:
    """Return the laser state.

    Returns:
        dict: ok() with is_on, power_armed, power_mw and the beam emitted if turned on, or fail().
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    try:
        st = _beam_state()
        armed, last = st["power_armed"], st["power_mw"]

        out = ok(is_on=bool(getattr(_hw._laser, "is_on", False)),
                 power_armed=armed,
                 power_mw=last if armed else None,
                 power_pct=st["power_pct"] if armed else None,
                 beam_if_turned_on=st["beam"])

        if not armed:
            out["last_requested_power_mw"] = last
            out["note"] = (
                "Power is NOT applied to the optics right now - the ND filter sits at the "
                "guide-beam/blocking position, so laser_on() would emit only the guide beam. "
                "Call set_laser_power(mw) to arm the measurement beam.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("set_guide_beam_mode")
def set_guide_beam_mode() -> dict:
    """Switch to guide-beam mode.

    Moves the beam splitter to standby and the ND filter to the blocking position.

    Returns:
        dict: ok() with a status message, or fail().
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    try:
        _hw._laser.set_guide_beam()
        return ok(status="Switched to guide-beam mode")
    except Exception as e:
        return fail(str(e))
