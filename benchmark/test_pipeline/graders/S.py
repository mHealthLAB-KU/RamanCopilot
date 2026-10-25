
from __future__ import annotations

from benchmark.test_pipeline.graders import _util as U

_DEVICES = ("stage", "laser", "ccd", "camera")


def S01(r):
    """List connected instruments; all four connection states must match reality."""
    yield r.called("get_hardware_status"), "queried get_hardware_status"
    conn = r.state("hw.connected") or {}
    for dev in _DEVICES:
        yield r.mentions(dev), f"{dev} included in the report"
        yield r.claims_negative(dev) is (conn.get(dev) is False), \
            f"{dev} connection matches reality ({conn.get(dev)})"


def S02(r):
    """Report the current X, Y, Z position without moving."""
    for ax in ("x", "y", "z"):
        yield r.cites(r.state(f"pos.{ax}"), tol=0.001), f"cited {ax} coordinate"
    yield r.unchanged("pos.x", "pos.y", "pos.z", tol=0.001), "read-only: stage position unchanged"


def S03(r):
    """Set the exposure to 2 s without measuring."""
    yield U.near(r.state("ccd.exposure_time_s"), 2.0), "exposure_time == 2.0 s"
    yield r.not_called("acquire_spectrum"), "did not acquire"


def S04(r):
    yield r.state("ccd.shutter_mode") == "close", "shutter == close"


def S05(r):
    """Combine four exposures into one (accumulate mode)."""
    yield r.state("ccd.acquisition_mode") == "accumulate", "acquisition_mode == accumulate"
    yield r.state("ccd.num_accumulations") == 4, "num_accumulations == 4"


def S06(r):
    yield r.state("ccd.acquisition_mode") == "kinetic", "acquisition_mode == kinetic"
    yield r.state("ccd.num_kinetics") == 20, "num_kinetics == 20"


def S07(r):
    """Single-track readout centered on row 50 with width 20 pixels."""
    yield r.state("ccd.read_mode") == "single_track", "read_mode == single_track"
    yield r.state("track.center") == 50, "single_track_center == 50 (driver attribute)"
    yield r.state("track.width") == 20, "single_track_width == 20 (driver attribute)"


def S08(r):
    yield r.state("ccd.read_mode") == "fvb", "read_mode == fvb"


def S09(r):
    yield r.state("ccd.trigger_mode") == "external", "trigger_mode == external"


def S10(r):
    """Report the detector pixel count."""
    yield r.cites(r.state("ccd.detector_Nx")), "cited detector Nx"
    yield r.cites(r.state("ccd.detector_Ny")), "cited detector Ny"


def S11(r):
    """Set X and Y speeds to 1 mm/s without touching Z."""
    yield U.near(r.state("speed.x_speed_mm_s"), 1.0), "x_speed == 1.0 mm/s"
    yield U.near(r.state("speed.y_speed_mm_s"), 1.0), "y_speed == 1.0 mm/s"
    yield r.unchanged("speed.z_speed_mm_s", tol=1e-6), "z_speed unchanged"


def S12(r):
    """Move to X=35, Y=25 keeping Z."""
    yield abs((r.state("pos.x") or 0) - 35.0) <= 0.001, "x == 35.000 mm"
    yield abs((r.state("pos.y") or 0) - 25.0) <= 0.001, "y == 25.000 mm"
    yield r.unchanged("pos.z", tol=0.001), "z unchanged"


def S13(r):
    """Move X by +0.05 mm relative; Y and Z unchanged."""
    yield r.moved("x", +0.05), "x displacement == +0.05 mm"
    yield r.unchanged("pos.y", "pos.z", tol=0.001), "y and z unchanged"


def S14(r):
    """Move 0.05 mm to the right on screen (sign convention)."""
    from backend.service.vision.optics_map import SIGN_X
    yield r.moved("x", SIGN_X * 0.05), f"x displacement == {SIGN_X * 0.05:+} mm (SIGN_X={SIGN_X})"
    yield r.unchanged("pos.y", tol=0.001), "y unchanged"


def S15(r):
    """Autofocus starting from Z=0.2; the final Z must be the returned optimum."""
    yield abs(float(r.args("run_autofocus").get("initial_z") or -99) - 0.2) < 1e-6, \
        "called with initial_z == 0.2"
    opt = r.result("run_autofocus").get("optimal_z")
    yield U.near(r.state("pos.z"), opt), \
        "final z == returned optimal_z"


def S16(r):
    yield r.ok("start_camera_stream"), "start_camera_stream succeeded"
    yield r.state("camera.streaming") is True, "is_streaming == True"


def S17(r):
    """Set the camera exposure to 20 ms (graded on call arguments)."""
    yield r.ok("set_camera_exposure"), "set_camera_exposure succeeded"
    yield float(r.args("set_camera_exposure").get("ms") or 0) == 20.0, "argument ms == 20"


def S18(r):
    """Turn auto exposure off (graded on call arguments)."""
    yield r.ok("set_camera_auto_exposure"), "set_camera_auto_exposure succeeded"
    yield r.args("set_camera_auto_exposure").get("enabled") is False, "argument enabled == False"


def S19(r):
    """Switch to guide-beam mode (graded on power_armed)."""
    yield r.state("laser.power_armed") is False, "power_armed == False (guide beam)"


def S20(r):
    yield r.state("laser.is_on") is False, "laser is_on == False"
