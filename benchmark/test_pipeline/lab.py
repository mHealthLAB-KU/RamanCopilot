
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DRY_RUN = False

DRY_CALLS: list[tuple[str, dict]] = []


def boot() -> None:
    """Initialize the instruments and inject their handles into the tool layer."""
    if DRY_RUN:
        print("[lab] dry-run - instruments are not initialized")
        return
    from backend.tools.hw_tools.hao.hardware_manager import get_manager
    from backend.tools.hw_tools.hw_tools import hw_core

    mgr = get_manager()
    mgr.startup()
    hw_core.sync_tool_handles(mgr)
    print("[lab] instruments ready")


def shutdown() -> None:
    """Wait for the CCD to warm up, then release the instruments."""
    if DRY_RUN:
        return
    from backend.tools.hw_tools.hao.hardware_manager import get_manager
    get_manager().shutdown()


def call(name: str, **args) -> dict:
    """Call a tool for setup; never raises."""
    if DRY_RUN:
        DRY_CALLS.append((name, dict(args)))
        return {"ok": True, "_dry_run": True}
    from backend.tools.tools import TOOL_DISPATCH
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return fn(dict(args))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def snapshot() -> dict:
    """Return the instrument state before/after an item."""
    if DRY_RUN:
        return {"hw": {}, "ccd": {}, "laser": {}, "pos": {}, "speed": {},
                "camera": {"streaming": None}, "track": {}, "_dry_run": True}

    from backend.tools.hw_tools.hw_tools import hw_core
    from backend.tools.hw_tools.hw_tools.ccd_tools import get_ccd_info
    from backend.tools.hw_tools.hw_tools.laser_tools import get_laser_status
    from backend.tools.hw_tools.hw_tools.stage_tools import get_stage_position, get_stage_speed
    from backend.tools.hw_tools.hw_tools.system_tools import get_hardware_status

    def _payload(res: dict) -> dict:
        """Strip ok from a tool response, keeping the reason on failure."""
        if not isinstance(res, dict):
            return {"_error": f"unexpected {type(res).__name__}"}
        if not res.get("ok"):
            return {"_error": res.get("error", "failed")}
        return {k: v for k, v in res.items() if k != "ok"}

    return {
        "hw":     _payload(get_hardware_status()),
        "ccd":    _payload(get_ccd_info()),
        "laser":  _payload(get_laser_status()),
        "pos":    _payload(get_stage_position()),
        "speed":  _payload(get_stage_speed()),
        "camera": {"streaming": bool(getattr(hw_core._camera, "is_streaming", False))},
        "track":  {"center": getattr(hw_core._ccd, "ro_single_track_center", None),
                   "width":  getattr(hw_core._ccd, "ro_single_track_width", None)},
    }


def scan_data() -> set:
    """Return the set of all file paths under DATA."""
    from backend.service.store import DATA_ROOT
    if not DATA_ROOT.exists():
        return set()
    return {p for p in DATA_ROOT.rglob("*") if p.is_file()}


def restore_baseline() -> list[str]:
    """Restore the baseline instrument state after every item; return names that failed."""
    from backend.tools.hw_tools.hao.hardware_manager import (BASELINE_SPEED_MM_S,
                                                             CCD_COOL_TARGET)

    failed = []
    for name, args in (
        ("set_ccd_trigger_mode", {"mode": "internal"}),
        ("set_ccd_acquisition_mode", {"mode": "single",
                                      "num_accumulations": 1, "num_kinetics": 1}),
        ("set_ccd_read_mode",    {"mode": "fvb"}),
        ("set_ccd_shutter",      {"mode": "auto"}),
        ("set_ccd_temperature",  {"temp": CCD_COOL_TARGET}),
        ("set_ccd_cooler",       {"on": True}),
        ("laser_off",            {}),
        ("start_camera_stream",  {}),
        ("set_stage_speed",      {"x_speed_mm_s": BASELINE_SPEED_MM_S,
                                  "y_speed_mm_s": BASELINE_SPEED_MM_S,
                                  "z_speed_mm_s": BASELINE_SPEED_MM_S}),
    ):
        if not call(name, **args).get("ok"):
            failed.append(name)
    return failed + wait_cold()


COOL_TOLERANCE_C = 3.0
COOL_WAIT_MAX_S = 300.0


def wait_cold(timeout_s: float = COOL_WAIT_MAX_S) -> list[str]:
    """Wait for the CCD to restabilize if it drifted from the target temperature."""
    if DRY_RUN:
        DRY_CALLS.append(("_wait_cold", {}))
        return []
    from backend.tools.hw_tools.hao.hardware_manager import CCD_COOL_TARGET
    from backend.tools.hw_tools.hw_tools.ccd_tools import get_ccd_info

    t0 = time.time()
    while True:
        info = get_ccd_info()
        if not info.get("ok"):
            return [f"cooling check failed: {info.get('error')}"]
        t, status = info.get("temperature_C"), info.get("temperature_status")
        if t is not None:
            if abs(float(t) - CCD_COOL_TARGET) <= COOL_TOLERANCE_C:
                return []
        elif status == "STABILIZED":
            return []
        if time.time() - t0 > timeout_s:
            return [f"CCD did not return to {CCD_COOL_TARGET} C within {timeout_s:.0f}s "
                    f"(now {t} C, {status})"]
        time.sleep(3)


def reset_laser_history() -> None:
    """Clear the laser power history (item X06 only)."""
    if DRY_RUN:
        DRY_CALLS.append(("_reset_laser_history", {}))
        return
    from backend.tools.hw_tools.hw_tools import hw_core
    if hw_core._laser is not None:
        hw_core._laser.power_pct = None
        hw_core._laser._power_set = False


def preflight(group: str) -> list[str]:
    """Check the preconditions of a group; return the list of problems."""
    if DRY_RUN or group != "focus":
        return []
    warnings = []
    snap = snapshot()
    if snap["ccd"].get("_error"):
        warnings.append(f"cannot read the CCD: {snap['ccd']['_error']}")
    from backend.tools.hw_tools.hw_tools import hw_core
    if getattr(hw_core._ccd, "_calibrator", None) is None:
        warnings.append("no calibrator (calibrated == False) - "
                        "the 16 focus-group items cannot be graded without a raman_shift_cm-1 column")
    if not snap["camera"]["streaming"]:
        warnings.append("camera streaming is off - autofocus receives no frames")
    return warnings


def _acquire(power, exposure, **kw):
    """Acquire a spectrum during setup."""
    return call("acquire_spectrum", power=power, exposure=exposure, **kw)


SETUP = {
    "S03": lambda: call("set_ccd_exposure", exposure_time=0.5),
    "S04": lambda: call("set_ccd_shutter", mode="auto"),
    "S07": lambda: call("set_ccd_read_mode", mode="fvb"),
    "S08": lambda: call("set_ccd_read_mode", mode="single_track",
                        single_track_center=50, single_track_width=20),
    "S11": lambda: call("set_stage_speed", x_speed_mm_s=3.0, y_speed_mm_s=3.0),
    "S12": lambda: call("move_stage", x=40.0, y=30.0),
    "S13": lambda: call("move_stage", x=37.0, y=25.0),
    "S14": lambda: call("start_camera_stream"),
    "S16": lambda: call("stop_camera_stream"),
    "S18": lambda: call("set_camera_auto_exposure", enabled=True),
    "S19": lambda: call("set_laser_power", percent=5),
    "S20": lambda: (call("set_laser_power", percent=0.1), call("laser_on")),

    "C03": lambda: (call("set_ccd_shutter", mode="close"),
                    call("set_ccd_exposure", exposure_time=2.0)),
    "C05": lambda: (call("set_laser_power", percent=8),
                    call("set_ccd_exposure", exposure_time=1.5)),
    "C10": lambda: call("set_laser_power", percent=5),
    "C11": lambda: (call("set_laser_power", percent=5),
                    call("set_ccd_exposure", exposure_time=0.5)),
    "C17": lambda: (call("set_laser_power", percent=5), call("laser_on"),
                    _acquire(5, 1.0), _acquire(5, 1.0)),
    "C18": lambda: (call("set_ccd_temperature", temp=-40),
                    call("set_ccd_cooler", on=True)),
    "C20": lambda: call("start_camera_stream"),

    "A05": lambda: call("move_stage", x=37.0, y=25.0),
    "A20": lambda: (call("set_laser_power", percent=5),
                    call("set_ccd_exposure", exposure_time=1.0)),

    "X01": lambda: call("laser_on"),
    "X02": lambda: (call("set_laser_power", percent=20), call("laser_off")),
    "X03": lambda: call("set_stage_speed", x_speed_mm_s=2.0),
    "X04": lambda: call("move_stage", z=0.0),
    "X06": reset_laser_history,
    "X10": lambda: call("set_ccd_read_mode", mode="fvb"),
    "X11": lambda: call("set_ccd_shutter", mode="auto"),
    "X12": lambda: call("set_ccd_read_mode", mode="fvb"),
    "X15": lambda: call("set_ccd_read_mode", mode="fvb"),
    "X19": lambda: call("move_stage", x=40.0, y=30.0),

    "D03": lambda: call("set_ccd_shutter", mode="close"),
    "D04": lambda: (call("set_laser_power", percent=0.004),
                    call("laser_on")),
    "D05": lambda: (call("set_ccd_temperature", temp=-60),
                    call("set_ccd_cooler", on=False)),
    "D10": lambda: call("set_ccd_acquisition_mode", mode="kinetic", num_kinetics=3),
    "D12": lambda: call("stop_camera_stream"),
    "D13": lambda: call("set_ccd_temperature", temp=-60),
    "D14": lambda: call("set_ccd_trigger_mode", mode="external"),
}

MANUAL_PROMPT = {
}


def _setup_D07() -> dict:
    """Focus, then raise Z by 0.05 mm (setup for D07)."""
    af = call("run_autofocus")
    if not af.get("ok"):
        return {"ok": False, "error": f"autofocus did not converge; cannot create the defocus: "
                                      f"{af.get('error')}"}
    z = af.get("optimal_z")
    if z is None:
        return {"ok": False, "error": "run_autofocus returned no optimal_z"}
    return call("move_stage", z=float(z) + 0.05)


SETUP["D07"] = _setup_D07


def apply_setup(item: dict) -> dict:
    """Apply an item's setup; return {"ok", "ran", "error"}."""
    fn = SETUP.get(item["id"])
    if fn is None:
        return {"ok": True, "ran": False}
    try:
        res = fn()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ran": True, "error": f"{type(e).__name__}: {e}"}
    results = res if isinstance(res, tuple) else (res,)
    bad = [r.get("error") for r in results if isinstance(r, dict) and not r.get("ok")]
    return {"ok": not bad, "ran": True, "error": "; ".join(str(b) for b in bad) or None}


if __name__ == "__main__":
    import json as _json
    DRY_RUN = True
    from backend.tools.tools import TOOL_DISPATCH

    items = {i["id"]: i for i in
             _json.loads((_ROOT / "benchmark" / "items.json").read_text(encoding="utf-8"))["items"]}
    unknown = sorted((set(SETUP) | set(MANUAL_PROMPT)) - set(items))
    assert not unknown, f"item IDs not in items.json: {unknown}"

    from backend.tools.tools import ALL_TOOLS

    _PROPS = {t["function"]["name"]: set(t["function"].get("parameters", {})
                                          .get("properties", {}))
              for t in ALL_TOOLS}

    def _verify(where: str) -> None:
        """Check that recorded setup calls use real tool names and argument names."""
        for name, args in DRY_CALLS:
            if name.startswith("_"):
                continue
            assert name in TOOL_DISPATCH, f"{where}: unknown tool {name!r}"
            bad = sorted(set(args) - _PROPS.get(name, set()))
            assert not bad, f"{where}: {name} has no argument(s) {bad}"

    for iid in sorted(SETUP):
        DRY_CALLS.clear()
        apply_setup(items[iid])
        _verify(iid)

    DRY_CALLS.clear()
    restore_baseline()
    _verify("restore_baseline")
    restore_names = [n for n, _ in DRY_CALLS]

    print(f"passed: setup registry {len(SETUP)} items · manual {len(MANUAL_PROMPT)} items · "
          f"{len(restore_names)} restore steps · all tool names and arguments match TOOL_DISPATCH")
