"""Status and settings for the instrument panel of the web interface."""

from __future__ import annotations

from backend.common import optics_map as _om
from backend.common.config.hw import CAMERA_HEIGHT, CAMERA_WIDTH, LENSES
from backend.common.result import fail, ok
from backend.repo.hw_access.hw_core import _STAGE_BUSY, _last_stage_values, _read_stage_values, _stage_read
from backend.repo.hw_access.hao.USE_stage_test import MAX_SPEED_XY, MAX_SPEED_Z, MIN_SPEED_MM_S
from backend.service.tools.hw_tools import camera_tools, ccd_tools, laser_tools, stage_tools
from backend.service.tools.hw_tools.hw_ops import LASER_FULL_MW, _RO_MODE_TO_ARG, pct_to_mw
from backend.service.tools.hw_tools.system_tools import ccd_temperature, connection_flags, select_objective


def _ccd_status(mgr) -> dict | None:
    ccd = mgr.ccd
    if ccd is None:
        return None
    a = lambda name, default=None: getattr(ccd, name, default)  # noqa: E731
    temp = ccd_temperature(mgr)
    chan = a("ad_chan", 0) or 0
    hs = a("HSSpeeds_Conventional") or []
    try:
        hbin = ccd.get_current_hbin()
    except Exception:  # noqa: BLE001
        hbin = None
    return {
        "temperature": temp["temperature"],
        "temp_status": temp["temp_status"],
        "target_temp": a("temperature_setpoint"),
        "cooler": a("cooler_on"),
        "exposure_s": a("exposure_time"),
        "acq_mode": a("aq_mode"),
        "num_acc": a("num_acc"),
        "num_kin": a("num_kin"),
        "kinetic_cycle_s": a("kinetic_cycle_time"),
        "read_mode": _RO_MODE_TO_ARG.get(a("ro_mode", ""), "fvb"),
        "hbin": hbin,
        "track_center": a("ro_single_track_center"),
        "track_width": a("ro_single_track_width"),
        "shutter": a("shutter_mode"),
        "preamp_index": a("preamp_gain_i"),
        "preamp_gains": list(a("preamp_gains") or []),
        "vs_index": a("vs_speed_index"),
        "vs_speeds_us": list(a("VSSpeeds") or []),
        "hs_index": a("hs_speed_index"),
        "hs_speeds_mhz": list(hs[chan]) if len(hs) > chan else [],
    }


def _laser_status(mgr) -> dict | None:
    laser = mgr.laser
    if laser is None:
        return None
    armed = bool(getattr(laser, "_power_set", False))
    return {
        "is_on": bool(getattr(laser, "is_on", False)),
        "beam": "measurement" if armed else "guide",
        "power_mw": pct_to_mw(getattr(laser, "power_pct", None)),
        "power_pct": None if getattr(laser, "power_pct", None) is None else round(float(laser.power_pct), 6),
        "full_mw": LASER_FULL_MW,
        "power_range": [pct_to_mw(getattr(laser, "ND_MIN_PCT", 0.004)), pct_to_mw(getattr(laser, "ND_MAX_PCT", 100.0))],
    }


def _camera_status(mgr) -> dict | None:
    cam = mgr.camera
    if cam is None:
        return None
    return {"streaming": bool(getattr(cam, "is_streaming", False)),
            "exposure_ms": getattr(cam, "exposure_ms", None),
            "auto_exposure": getattr(cam, "auto_exposure", None)}


def _stage_status(mgr) -> dict | None:
    stage = mgr.stage
    if stage is None:
        return None
    snap = _stage_read(lambda: _read_stage_values(stage))
    if snap is _STAGE_BUSY:
        snap = _last_stage_values.get("value", {})
    else:
        _last_stage_values["value"] = snap
    out = {k: snap[k] for k in ("x", "y", "z") if k in snap}
    if "velocity" in snap:
        out["speed"] = snap["velocity"]
    out["speed_range"] = {"x": [MIN_SPEED_MM_S, MAX_SPEED_XY], "y": [MIN_SPEED_MM_S, MAX_SPEED_XY],
                          "z": [MIN_SPEED_MM_S, MAX_SPEED_Z]}
    return out


def _optics_status() -> dict:
    aim_x, aim_y = _om.aim_px()
    fov_x, fov_y = _om.fov_mm()
    return {"objective": _om.objective(), "available": list(LENSES),
            "aim": {"x": round(aim_x, 1), "y": round(aim_y, 1), "width": CAMERA_WIDTH, "height": CAMERA_HEIGHT},
            "fov_mm": {"x": round(fov_x, 4), "y": round(fov_y, 4)}}


def hardware_status(mgr) -> dict:
    """Return the full hardware status for the panel (None for disconnected instruments)."""
    return {
        "connected": connection_flags(mgr),
        "ccd": _ccd_status(mgr),
        "laser": _laser_status(mgr),
        "camera": _camera_status(mgr),
        "stage": _stage_status(mgr),
        "optics": _optics_status(),
    }


def _ccd_calls(c: dict, mgr) -> list[tuple[str, callable]]:
    """Convert CCD changes into tool calls."""
    if not c:
        return []
    calls = []
    ccd = mgr.ccd
    if "exposure_s" in c:
        calls.append(("ccd.exposure_s", lambda: ccd_tools.set_ccd_exposure(c["exposure_s"])))
    if {"acq_mode", "num_acc", "num_kin", "kinetic_cycle_s"} & c.keys():
        calls.append(("ccd.acq_mode", lambda: ccd_tools.set_ccd_acquisition_mode(
            c.get("acq_mode") or getattr(ccd, "aq_mode", "single"),
            num_accumulations=c.get("num_acc"), num_kinetics=c.get("num_kin"),
            kinetic_cycle_time=c.get("kinetic_cycle_s"))))
    if {"read_mode", "hbin", "track_center", "track_width"} & c.keys():
        calls.append(("ccd.read_mode", lambda: ccd_tools.set_ccd_read_mode(
            c.get("read_mode") or _RO_MODE_TO_ARG.get(getattr(ccd, "ro_mode", ""), "fvb"),
            hbin=c.get("hbin"), single_track_center=c.get("track_center"),
            single_track_width=c.get("track_width"))))
    if "shutter" in c:
        calls.append(("ccd.shutter", lambda: ccd_tools.set_ccd_shutter(c["shutter"])))
    if "preamp_index" in c:
        calls.append(("ccd.preamp_index", lambda: ccd_tools.set_ccd_preamp_gain(c["preamp_index"])))
    if {"vs_index", "hs_index"} & c.keys():
        calls.append(("ccd.speeds", lambda: ccd_tools.set_ccd_shift_speeds(
            vs_index=c.get("vs_index"), hs_index=c.get("hs_index"))))
    if "target_temp" in c:
        calls.append(("ccd.target_temp", lambda: ccd_tools.set_ccd_temperature(c["target_temp"])))
    if "cooler" in c:
        calls.append(("ccd.cooler", lambda: ccd_tools.set_ccd_cooler(c["cooler"])))
    return calls


def apply_hw_settings(changes: dict, mgr) -> dict:
    """Apply only the fields changed in the panel by calling the corresponding tools."""
    calls: list[tuple[str, callable]] = []
    calls += _ccd_calls(changes.get("ccd") or {}, mgr)

    laser = changes.get("laser") or {}
    if "power_mw" in laser:
        calls.append(("laser.power_mw", lambda: laser_tools.set_laser_power(laser["power_mw"])))

    cam = changes.get("camera") or {}
    if "auto_exposure" in cam:
        calls.append(("camera.auto_exposure", lambda: camera_tools.set_camera_auto_exposure(cam["auto_exposure"])))
    if "exposure_ms" in cam:
        calls.append(("camera.exposure_ms", lambda: camera_tools.set_camera_exposure(cam["exposure_ms"])))

    speed = (changes.get("stage") or {}).get("speed") or {}
    if speed:
        calls.append(("stage.speed", lambda: stage_tools.set_stage_speed(
            x_speed_mm_s=speed.get("x"), y_speed_mm_s=speed.get("y"), z_speed_mm_s=speed.get("z"))))

    objective = (changes.get("optics") or {}).get("objective")
    if objective:
        calls.append(("optics.objective", lambda: select_objective(objective)))

    if not calls:
        return fail("Nothing to change.")

    applied, errors, busy = [], {}, None
    for name, call in calls:
        res = call()
        if res.get("ok"):
            applied.append(name)
        else:
            errors[name] = res.get("error", "failed")
            busy = busy or res.get("busy_with")
    if errors:
        out = fail(" · ".join(f"{k}: {v}" for k, v in errors.items()), errors=errors, applied=applied)
        if busy:
            out["busy_with"] = busy
        return out
    return ok(applied=applied)
