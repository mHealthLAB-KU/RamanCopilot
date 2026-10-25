"""Common paths that apply laser power, guide-beam state and CCD modes."""

from __future__ import annotations

import time

from backend.common.config.hw import LASER_FULL_MW
from backend.common.result import fail
from backend.repo.hw_access import hw_core as _hw
from backend.service.tools.hw_tools.guards import _check_ccd_positive


_ND_SETTLE_S = 0.15  # wait after the ND filter motor stops


def pct_to_mw(pct):
    """Convert ND transmission (%) to mW."""
    return None if pct is None else round(float(pct) * LASER_FULL_MW / 100.0, 6)


def mw_to_pct(mw: float) -> float:
    """Convert mW to ND transmission (%)."""
    return float(mw) * 100.0 / LASER_FULL_MW


POWER_MIN_MW, POWER_MAX_MW = pct_to_mw(0.004), LASER_FULL_MW
POWER_RANGE_TXT = f"{POWER_MIN_MW:g}-{POWER_MAX_MW:g} mW ({POWER_MAX_MW:g} mW = full power)"


def _laser_power_range() -> tuple[float, float]:
    """Return the allowed laser power range (mW)."""
    return (pct_to_mw(getattr(_hw._laser, "ND_MIN_PCT", 0.004)),
            pct_to_mw(getattr(_hw._laser, "ND_MAX_PCT", 100.0)))


def _apply_laser_power(mw, settle_s: float = _ND_SETTLE_S) -> dict | None:
    """Validate and apply the laser power (mW); return an error dict or None."""
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    if isinstance(mw, bool) or not isinstance(mw, (int, float)):
        try:
            mw = float(mw)
        except (TypeError, ValueError):
            return fail("power must be a number (mW).")
    lo, hi = _laser_power_range()
    if not (lo <= float(mw) <= hi):
        return fail(f"Valid power range: {lo} - {hi} mW")
    try:
        pct = min(max(mw_to_pct(mw), float(getattr(_hw._laser, "ND_MIN_PCT", 0.004))),
                  float(getattr(_hw._laser, "ND_MAX_PCT", 100.0)))
        applied = _hw._laser.set_power(pct)
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    time.sleep(settle_s)
    if applied is False:
        return fail("The ND filter motor did not confirm the move, so the laser power was NOT applied. "
                    "The measurement beam is still un-armed. Retry, or check the laser controller link.")
    return None


def _beam_state() -> dict:
    """Return which beam laser_on() would emit now."""
    armed = bool(getattr(_hw._laser, "_power_set", False))
    pct = getattr(_hw._laser, "power_pct", None)
    return {"power_armed": armed,
            "power_mw": pct_to_mw(pct),
            "power_pct": None if pct is None else round(float(pct), 6),
            "beam": "measurement" if armed else "guide"}


def _laser_off_quiet() -> bool:
    """Turn the laser off without raising (for finally blocks), blocking the beam if needed."""
    if _hw._laser is None:
        return True
    try:
        if _hw._laser.laser_off() is not False:
            return True
    except Exception:
        pass
    print("[laser] SSPW 0 unconfirmed - falling back to ND blocking.")
    return _restore_guide_beam_quiet()


def _restore_guide_beam_quiet() -> None:
    """Return the optics to the guide-beam (camera) position without raising."""
    if _hw._laser is None:
        return False
    try:
        _hw._laser.set_guide_beam()
        return True
    except Exception as e:
        print(f"[laser] guide-beam restore failed: {type(e).__name__}: {e}")
        return False


_ACQ_MODES      = ('single', 'accumulate', 'kinetic', 'run_till_abort')
_ACQ_MODES_1D   = ('single', 'accumulate', 'kinetic')
_READ_MODES     = ('fvb', 'single_track', 'image')
_READ_MODES_1D  = ('fvb', 'single_track')
_SHUTTER_MODES  = ('auto', 'open', 'close')

_RO_MODE_TO_ARG = {'FULL_VERTICAL_BINNING': 'fvb', 'SINGLE_TRACK': 'single_track', 'IMG': 'image'}


def _current_read_mode() -> str:
    """Return the current CCD read mode name."""
    return _RO_MODE_TO_ARG.get(getattr(_hw._ccd, 'ro_mode', ''), 'fvb')


def _apply_acq_mode(mode: str, exposure=None, num_accumulations=None,
                    kinetic_count=None, kinetic_cycle_time=None) -> dict | None:
    """Apply the acquisition mode and its parameters (call before _apply_read_mode)."""
    if mode not in _ACQ_MODES:
        return fail(f"acquisition mode must be one of {list(_ACQ_MODES)}.")
    for _n, _v, _i in (("exposure", exposure, False),
                       ("num_accumulations", num_accumulations, True),
                       ("kinetic_count", kinetic_count, True),
                       ("kinetic_cycle_time", kinetic_cycle_time, False)):
        err = _check_ccd_positive(_n, _v, integer=_i)
        if err:
            return err
    if mode == 'single':
        _hw._ccd.set_aq_single_scan(exposure=exposure)
    elif mode == 'accumulate':
        _hw._ccd.set_aq_accumulate_scan(exposure_time=exposure, num_acc=num_accumulations,
                                        cycle_time=0.1)
    elif mode == 'kinetic':
        _hw._ccd.set_aq_kinetic_scan(
            exp_time=exposure,
            num_kin=kinetic_count,
            num_acc=num_accumulations if (num_accumulations or 0) >= 1 else None,
            acc_time=0.1,
            kin_time=kinetic_cycle_time if kinetic_cycle_time is not None else 0.1,
        )
    else:
        _hw._ccd.set_aq_run_till_abort_scan()
        if exposure is not None:
            _hw._ccd.set_exposure_time(exposure)
    if num_accumulations is not None and mode in ('single', 'run_till_abort'):
        _hw._ccd.set_num_accumulations(num_accumulations)
    if kinetic_count is not None and mode != 'kinetic':
        _hw._ccd.set_num_kinetics(kinetic_count)
    return None


def _apply_read_mode(mode: str, hbin=None, single_track_center=None,
                     single_track_width=None) -> dict | None:
    """Apply the read mode; return an error dict or None."""
    if mode not in _READ_MODES:
        return fail(f"read mode must be one of {list(_READ_MODES)}.")
    for _n, _v in (("hbin", hbin), ("single_track_center", single_track_center),
                   ("single_track_width", single_track_width)):
        err = _check_ccd_positive(_n, _v, integer=True)
        if err:
            return err
    eff_hbin = hbin if hbin is not None else _hw._ccd.get_current_hbin()
    if mode == 'fvb':
        _hw._ccd.set_ro_full_vertical_binning(hbin=eff_hbin)
    elif mode == 'single_track':
        center = (single_track_center if single_track_center is not None
                  else getattr(_hw._ccd, 'ro_single_track_center', None))
        if center is None:
            return fail("single_track mode requires the single_track_center parameter.")
        _hw._ccd.set_ro_single_track(
            center=center,
            width=(single_track_width if single_track_width is not None
                   else getattr(_hw._ccd, 'ro_single_track_width', 1)),
            hbin=eff_hbin,
        )
    else:
        _hw._ccd.set_ro_image_mode(hbin=eff_hbin)
    return None


def _apply_shutter(mode: str, explicit: bool = False) -> dict | None:
    """Apply the shutter mode; return an error dict or None."""
    if mode not in _SHUTTER_MODES:
        return fail(f"shutter must be one of {list(_SHUTTER_MODES)}.")
    if mode == 'auto':
        _hw._ccd.set_shutter_auto()
    elif mode == 'open':
        _hw._ccd.set_shutter_open(True)
    else:
        _hw._ccd.set_shutter_close()
    if explicit:
        try:
            _hw._ccd.shutter_explicit = True
        except Exception:
            pass
    return None


def _effective_shutter(requested: str | None) -> str:
    """Decide the shutter mode acquire_spectrum will actually use."""
    if requested is not None:
        return requested
    if getattr(_hw._ccd, 'shutter_explicit', False):
        return getattr(_hw._ccd, 'shutter_mode', None) or 'auto'
    return 'auto'
