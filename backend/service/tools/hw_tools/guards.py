"""Checks shared by instrument tools before they touch the hardware."""

from __future__ import annotations

import functools

from backend.common.config.hw import STAGE_MAX_X, STAGE_MAX_Y, STAGE_MAX_Z, STAGE_MIN_Z
from backend.common.result import fail
from backend.repo.hw_access import hw_core as _hw

def _serialized(what: str, timeout: float = _hw._BUSY_TIMEOUT_S):
    """Decorator that serializes instrument-moving tools and returns an error dict on lock timeout."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _hw._INSTRUMENT_LOCK.acquire(timeout=timeout):
                busy = _hw._lock_holder.get("what") or "another operation"
                return fail(f"The instrument is busy with '{busy}' and did not become free within "
                            f"{timeout:.0f}s, so '{what}' was NOT performed. Only one operation may drive "
                            f"the hardware at a time (a measurement, a stage move, or a grid scan started "
                            f"from the parameter panel or another chat session). Nothing was changed - "
                            f"wait for it to finish and try again.",
                            busy_with=busy)
            prev = _hw._lock_holder.get("what")
            _hw._lock_holder["what"] = what if prev is None else prev
            try:
                return fn(*args, **kwargs)
            finally:
                _hw._lock_holder["what"] = prev
                _hw._INSTRUMENT_LOCK.release()
        return wrapper
    return deco


def _stage_unavailable() -> dict:
    """Return an error dict if the stage cannot be used, else None."""
    if _hw._stage is None:
        return fail("Stage is not initialized.")
    if getattr(_hw._stage, "dead", False):
        return fail(f"The stage session is dead and cannot be used: {getattr(_hw._stage, 'dead_reason', 'unknown')}. "
                    f"The DLL session could not be released, so no tool can revive it - the server process must "
                    f"be restarted. Do NOT retry reconnect_hardware; continue without the stage if the task "
                    f"allows it and say so explicitly in your answer.")
    return None


def _check_stage_target(x=None, y=None, z=None) -> dict | None:
    """Check that a stage target is within range; return an error dict or None."""
    if x is not None and not (0 <= float(x) <= STAGE_MAX_X):
        return fail(f"X out of range: {x} (allowed: 0-{STAGE_MAX_X})")
    if y is not None and not (0 <= float(y) <= STAGE_MAX_Y):
        return fail(f"Y out of range: {y} (allowed: 0-{STAGE_MAX_Y})")
    if z is not None and not (STAGE_MIN_Z <= float(z) <= STAGE_MAX_Z):
        return fail(f"Z out of range: {z} (allowed: {STAGE_MIN_Z}-{STAGE_MAX_Z})")
    return None

_CCD_NOT_READY = ("The spectrometer (CCD) is not connected yet - it is still initialising (a few "
                  "seconds after start-up) or failed to connect. Check get_hardware_status(); "
                  "if it stays missing, try reconnect_hardware(component='ccd').")


def _ccd_ready() -> dict | None:
    """Return an error dict if the CCD is not available, else None."""
    return None if _hw._ccd is not None else fail(_CCD_NOT_READY)


def _check_ccd_positive(name: str, value, integer: bool = False) -> dict | None:
    """Check that CCD numeric parameters are positive; return an error dict or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fail(f"{name} must be a number, got {value!r}.")
    if integer:
        if int(value) != value:
            return fail(f"{name} must be a whole number, got {value!r}.")
        if int(value) < 1:
            return fail(f"{name} must be at least 1, got {value!r}. Omit the parameter to keep the "
                        f"instrument's current setting - it is not a way to switch the feature off.")
    elif float(value) <= 0:
        return fail(f"{name} must be greater than 0, got {value!r}. Omit the parameter to keep the "
                    f"instrument's current setting.")
    return None
