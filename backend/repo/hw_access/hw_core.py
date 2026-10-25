"""Shared instrument handles and the lock that serializes instrument access."""

from __future__ import annotations


_stage = None
_laser = None
_ccd   = None
_camera = None


import contextlib
import threading

_INSTRUMENT_LOCK = threading.RLock()

_BUSY_TIMEOUT_S = 30.0  # maximum wait for the instrument lock

_lock_holder: dict = {"what": None}


class InstrumentBusy(RuntimeError):
    """Raised when the instrument lock cannot be acquired in time."""


@contextlib.contextmanager
def instrument_guard(what: str = "hardware connect", timeout: float = None):
    """Context manager that serializes instrument operations outside the tools (lock order: instrument_guard, then component_lock)."""
    t = _BUSY_TIMEOUT_S if timeout is None else timeout
    if not _INSTRUMENT_LOCK.acquire(timeout=t):
        busy = _lock_holder.get("what") or "another operation"
        raise InstrumentBusy(
            f"The instrument is busy with '{busy}' and did not become free within {t:.0f}s, "
            f"so '{what}' was NOT performed. Wait for the running operation to finish and retry.")
    prev = _lock_holder.get("what")
    _lock_holder["what"] = what if prev is None else prev
    try:
        yield
    finally:
        _lock_holder["what"] = prev
        _INSTRUMENT_LOCK.release()


_STAGE_BUSY = object()


def _stage_read(fn):
    """Run a stage DLL read under component_lock('stage'); return _STAGE_BUSY if the lock is unavailable."""
    try:
        from backend.repo.hw_access.hao.hardware_manager import get_manager
        lock = get_manager().component_lock("stage")
    except Exception:
        return fn()

    if not lock.acquire(blocking=False):
        return _STAGE_BUSY
    try:
        return fn()
    finally:
        lock.release()


def init_hardware(stage=None, laser=None, ccd=None, camera=None):
    """Inject the hardware handles used by the tools."""
    global _stage, _laser, _ccd, _camera
    got = _INSTRUMENT_LOCK.acquire(timeout=_BUSY_TIMEOUT_S)
    if not got:
        print(f"[WARN] hw_core.init_hardware(): could not take the instrument lock within "
              f"{_BUSY_TIMEOUT_S:.0f}s because '{_lock_holder.get('what')}' is running - "
              f"replacing the handles anyway.")
    try:
        _stage = stage
        _laser = laser
        _ccd = ccd
        _camera = camera
    finally:
        if got:
            _INSTRUMENT_LOCK.release()
    print(f"[DEBUG] hw_core.init_hardware() called: stage={_stage}, laser={_laser}, "
          f"ccd={_ccd}, camera={_camera}")


_last_stage_values: dict = {"value": {}}


def _read_stage_values(stage) -> dict:
    """Read stage position and velocity in frontend format (call only inside _stage_read)."""
    out: dict = {}
    try:
        pos = stage.get_position()
        out = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
    except Exception:
        out = {}
    try:
        vel = stage.get_velocity()
        if isinstance(vel, dict) and vel.get("ok"):
            out["velocity"] = {"x": float(vel["x_speed_mm_s"]),
                               "y": float(vel["y_speed_mm_s"]),
                               "z": float(vel["z_speed_mm_s"])}
    except Exception:
        pass
    return out


def sync_tool_handles(mgr) -> None:
    """Re-inject the manager's current handles into this module after a (re)connection."""
    init_hardware(stage=getattr(mgr, "stage", None), laser=getattr(mgr, "laser", None),
                  ccd=getattr(mgr, "ccd", None), camera=getattr(mgr, "camera", None))
