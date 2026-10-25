"""Agent tools for instrument status and reconnection."""

from __future__ import annotations

import time

from backend.common.result import fail, ok
from pydantic import Field
from typing import Annotated, Literal, Optional
from backend.common.config import virtual_hw as vhw_cfg
from backend.common import optics_map as _om
from backend.common.config.hw import LENSES
from backend.service.tools.hw_tools.guards import _serialized
from backend.repo.hw_access import hw_core as _hw
from backend.repo.hw_access.hw_core import InstrumentBusy, _STAGE_BUSY, _last_stage_values, _read_stage_values, _stage_read, instrument_guard, sync_tool_handles


_RECONNECT_SETTLE_S = 1.5  # wait between release and reconnect
_RECONNECT_ATTEMPTS = 2  # initialization attempts per reconnect


def _teardown_component(mgr, comp: str) -> dict:
    """Release one component before reconnecting and report each step's success or failure."""
    steps: dict = {}

    def _try(name, fn) -> bool:
        """Run one teardown step; an exception or a False return counts as failure."""
        try:
            r = fn()
        except Exception as e:
            steps[name] = f"{type(e).__name__}: {e}"
            return False
        if r is False:
            steps[name] = "returned False (device reported failure)"
            return False
        steps[name] = "ok"
        return True

    obj = getattr(mgr, comp, None)
    if obj is None:
        return {"released": True, "steps": {"skip": "was not connected"}}

    if comp == "stage":
        _try("disconnect", obj.disconnect)
        freed = _try("free_session", obj.free_session)
        released = freed
        if not released and hasattr(obj, "mark_dead"):
            obj.mark_dead(f"release failed during reconnect ({steps})")
    elif comp == "camera":
        _try("stop_stream", obj.stop_stream)
        released = _try("close", obj.close)
    elif comp == "ccd":
        released = _try("close", obj.close)
    elif comp == "laser":
        _try("laser_off", obj.laser_off)

        def _close_serial():
            ser = getattr(obj, "ser", None)
            if ser is None:
                return None
            if getattr(ser, "is_open", False):
                ser.close()
            return not getattr(ser, "is_open", False)

        released = _try("close_serial", _close_serial)
    else:
        return {"released": False, "steps": {"error": f"unknown component '{comp}'"}}

    if released:
        setattr(mgr, comp, None)
    return {"released": released, "steps": steps}


def get_hardware_status() -> dict:
    """Return which instruments are connected and the reason for any that are not.

    Returns:
        dict: ok() with the state of each instrument, or fail().
    """
    try:
        from backend.repo.hw_access.hao.hardware_manager import get_manager
        mgr = get_manager()
    except Exception as e:
        return fail(f"HardwareManager unavailable: {type(e).__name__}: {e}")

    out: dict = ok(connected={}, notes={})
    for name in ("stage", "laser", "ccd", "camera"):
        obj = getattr(mgr, name, None)
        dead = obj is not None and getattr(obj, "dead", False)
        out["connected"][name] = obj is not None and not dead
        if obj is None:
            out["notes"][name] = ("Not connected. Try reconnect_hardware(component='%s'). "
                                  "If that reports the resource is still held, the server process "
                                  "must be restarted - no tool can clear it." % name)
        elif dead:
            out["notes"][name] = (
                "Dead session (%s): the handle still exists but the DLL session could not be "
                "released, so no tool can revive it. reconnect_hardware will NOT help - the server "
                "process must be restarted. Continue without the %s if the task allows it."
                % (getattr(obj, "dead_reason", "unknown"), name))

    if out["connected"].get("stage"):
        try:
            probe = _stage_read(lambda: mgr.stage.get_position())
            if probe is _STAGE_BUSY:
                out["notes"]["stage"] = ("The stage is being connected or released by another thread "
                                         "right now, so its position was not read. Nothing is wrong - "
                                         "check again in a few seconds.")
            else:
                out["stage_position"] = (
                    {"x": probe[0], "y": probe[1], "z": probe[2]} if probe else None)
                if not probe:
                    out["notes"]["stage"] = ("Handle exists but get_position() returned nothing - "
                                             "stage is unresponsive.")
        except Exception as e:
            out["notes"]["stage"] = f"Handle exists but reading position raised {type(e).__name__}: {e}"
    if mgr.laser is not None:
        try:
            ser = getattr(mgr.laser, "ser", None)
            out["laser_serial_open"] = bool(getattr(ser, "is_open", False)) if ser else None
        except Exception as e:
            out["notes"]["laser"] = f"{type(e).__name__}: {e}"

    ready = [k for k, v in out["connected"].items() if v]
    out["summary"] = (f"connected: {', '.join(ready) if ready else 'none'}; "
                      f"missing: {', '.join(k for k, v in out['connected'].items() if not v) or 'none'}")
    return out


@_serialized("select_objective")
def select_objective(objective: str) -> dict:
    """Set the mounted objective and restore its laser spot position.

    Args:
        objective (str): Objective name, for example 50x.

    Returns:
        dict: ok() with the objective, available objectives and field of view, or fail().
    """
    try:
        _om.set_objective(objective)
    except ValueError as e:
        return fail(str(e), available=list(LENSES))
    from backend.repo.hw_access.hao.hardware_manager import restore_laser_aim
    restore_laser_aim()
    fov_x, fov_y = _om.fov_mm()
    return ok(objective=_om.objective(), available=list(LENSES),
              fov_mm={"x": round(fov_x, 4), "y": round(fov_y, 4)})


def reconnect_hardware(
    component: Annotated[Optional[Literal['stage', 'ccd', 'camera', 'laser', 'all']], Field(description="Which component to reconnect. Default 'all'.")] = 'all',
) -> dict:
    """Release and initialize again one instrument or all of them.

    Returns:
        dict: ok() with the reconnected components, or fail() with the errors.
    """
    try:
        from backend.repo.hw_access.hao.hardware_manager import get_manager
    except Exception as e:
        return fail(f"HardwareManager import failed: {e}")

    comp = str(component or "all").strip().lower()
    valid = {"stage", "ccd", "camera", "laser", "all"}
    if comp not in valid:
        return fail(f"component must be one of {sorted(valid)}")

    try:
        mgr = get_manager()
    except Exception as e:
        return fail(f"HardwareManager unavailable: {type(e).__name__}: {e}")

    targets = ["stage", "ccd", "camera", "laser"] if comp == "all" else [comp]
    done, errors, detail = [], {}, {}

    try:
        _guard = instrument_guard(f"reconnect_hardware({comp})")
        _guard.__enter__()
    except InstrumentBusy as e:
        return fail(str(e),
                    reconnected=[],
                    detail={"skipped": "instrument busy"},
                    hint="A measurement or scan is still running. Nothing was changed - "
                         "wait for it to finish (check get_hardware_status()) and retry.")

    try:
        return _reconnect_locked(mgr, targets, done, errors, detail)
    finally:
        _guard.__exit__(None, None, None)


def _reconnect_locked(mgr, targets, done, errors, detail) -> dict:
    """Body of reconnect_hardware; called only while holding the instrument lock."""
    for t in targets:
        lock = mgr.component_lock(t)
        if not lock.acquire(blocking=False):
            detail[t] = {"skipped": "component busy (lock held by another thread)"}
            errors[t] = (
                f"The {t} is being connected or released by another thread right now (the "
                f"server's own reconnect endpoint, or the CCD cooling thread which runs for "
                f"several minutes). Reconnecting was NOT attempted: two threads initializing the "
                f"same device at once can orphan the handle and force a server restart. Do NOT "
                f"retry immediately - continue with the remaining hardware, then check "
                f"get_hardware_status() to see whether it came back on its own."
            )
            continue

        try:
            td = _teardown_component(mgr, t)
            detail[t] = {"teardown": td}

            if not td["released"]:
                errors[t] = (
                    f"Could not release the {t} before reconnecting - the resource is still held "
                    f"by this process (teardown steps: {td['steps']}). Reconnecting was NOT attempted, "
                    f"because doing so would fail with an 'already in use' error and could orphan the "
                    f"handle. This is a process-level lock, not a cable problem: retrying this tool will "
                    f"not clear it. Continue with the remaining hardware if possible, report this in your "
                    f"final answer, and note that the server process must be restarted to free the {t}."
                )
                continue

            time.sleep(_RECONNECT_SETTLE_S)

            last = None
            for attempt in range(1, _RECONNECT_ATTEMPTS + 1):
                try:
                    getattr(mgr, f"_init_{t}")()
                    done.append(t)
                    detail[t]["init"] = f"ok (attempt {attempt})"
                    last = None
                    break
                except Exception as e:
                    last = f"{type(e).__name__}: {e}"
                    detail[t]["init"] = f"attempt {attempt} failed: {last}"
                    if attempt < _RECONNECT_ATTEMPTS:
                        time.sleep(_RECONNECT_SETTLE_S)
            if last is not None:
                errors[t] = (
                    f"Released the {t} successfully but re-initialization failed after "
                    f"{_RECONNECT_ATTEMPTS} attempts: {last}. The resource is now free, so this is a "
                    f"device-side problem (power, cable, driver, or the device is held by another "
                    f"program such as rays-on.exe) - not something you can fix by calling tools again. "
                    f"Proceed without the {t} if the task allows it and say so explicitly in your answer."
                )
        finally:
            lock.release()

    sync_tool_handles(mgr)
    now_connected = {k: getattr(mgr, k, None) is not None
                     for k in ("stage", "laser", "ccd", "camera")}

    if errors:
        return fail("; ".join(f"{k}: {v}" for k, v in errors.items()),
                    reconnected=done, detail=detail, now_connected=now_connected)
    return ok(reconnected=done, detail=detail, now_connected=now_connected)


class StageSessionDead(RuntimeError):
    """Raised when an unreleased stage session remains; the process must be restarted."""


def connect_camera(hw, exposure_ms) -> str:
    """Connect the microscope camera, or the virtual camera if RAMAN_VIRTUAL_HW=1.

    Args:
        hw (HardwareManager): Instrument manager.
        exposure_ms (float): Initial exposure time.

    Returns:
        str: Status message.
    """
    with instrument_guard("camera connect"), hw.component_lock("camera"):
        if hw.camera is not None:
            return "camera already connected"
        if vhw_cfg.VIRTUAL_HW:
            from backend.repo.hw_access.hao_vertual.virtual_camera import VirtualCamera
            from virtual_benchmark.rig import get_sample
            cam = VirtualCamera(hw, scene=get_sample(), exposure_ms=exposure_ms)
        else:
            from backend.repo.hw_access.hao.USE_camera_stream import StreamingTUCam
            cam = StreamingTUCam(exposure_ms=exposure_ms)
        cam.start_stream()
        hw.camera = cam
        sync_tool_handles(hw)
        return "camera connected"


def connect_stage(hw) -> str:
    """Connect the stage, or the virtual stage if RAMAN_VIRTUAL_HW=1.

    Args:
        hw (HardwareManager): Instrument manager.

    Returns:
        str: Status message.

    Raises:
        StageSessionDead: If the existing stage session cannot be released.
    """
    with instrument_guard("stage connect"), hw.component_lock("stage"):
        if hw.stage is not None:
            if getattr(hw.stage, "dead", False):
                raise StageSessionDead(
                    f"the stage session is invalid ({getattr(hw.stage, 'dead_reason', 'unknown')}). "
                    f"The DLL session could not be released, so reconnecting cannot recover it - "
                    f"restart the server process.")
            return "stage already connected"
        hw._init_stage()
        sync_tool_handles(hw)
        return "stage connected"


def connect_laser(hw, port) -> str:
    """Connect the laser, or the virtual laser if RAMAN_VIRTUAL_HW=1.

    Args:
        hw (HardwareManager): Instrument manager.
        port (str): Serial port.

    Returns:
        str: Status message.
    """
    with instrument_guard("laser connect"), hw.component_lock("laser"):
        if hw.laser is not None:
            return "laser already connected"
        if vhw_cfg.VIRTUAL_HW:
            from backend.repo.hw_access.hao_vertual.virtual_laser import VirtualLaser
            laser = VirtualLaser(port=port)
        else:
            from backend.repo.hw_access.hao.USE_laser_with_power import LaserController
            laser = LaserController(port=port)
        if not (laser.ser and laser.ser.is_open):
            raise RuntimeError(f"laser port connection failed ({port})")
        hw.laser = laser
        sync_tool_handles(hw)
        return "laser connected"


def connection_flags(hw) -> dict:
    """Return which instruments have a handle, without accessing the hardware.

    Args:
        hw (HardwareManager): Instrument manager.

    Returns:
        dict: True or False for stage, ccd, camera and laser.
    """
    return {"stage":  hw.stage  is not None,
            "ccd":    hw.ccd    is not None,
            "camera": hw.camera is not None,
            "laser":  hw.laser  is not None,
            "llm":    hw.ollama is not None}


def ccd_temperature(hw) -> dict:
    """Return the CCD temperature and cooler status.

    The last value is kept while the CCD is acquiring.

    Args:
        hw (HardwareManager): Instrument manager.

    Returns:
        dict: connected, temperature and temp_status.
    """
    ccd = hw.ccd
    if ccd is None:
        return {"connected": False, "temperature": None, "temp_status": None}
    try:
        try:
            t = ccd.get_temperature()
            s = ccd.temp_status_dict.get(ccd.temperature_status_num)
            return {"connected": True, "temperature": int(t), "temp_status": s}
        except IOError:
            cached_t = getattr(ccd, "temperature", None)
            cached_s = ccd.temp_status_dict.get(getattr(ccd, "temperature_status_num", None))
            return {"connected": True,
                    "temperature": int(cached_t) if cached_t is not None else None,
                    "temp_status": cached_s}
    except Exception:  # noqa: BLE001
        return {"connected": True, "temperature": None, "temp_status": None}


def latest_frame(hw):
    """Return the latest camera frame.

    Args:
        hw (HardwareManager): Instrument manager.

    Returns:
        np.ndarray or None: The frame, or None if no camera is connected.
    """
    cam = hw.camera
    return None if cam is None else cam.get_latest_frame()
