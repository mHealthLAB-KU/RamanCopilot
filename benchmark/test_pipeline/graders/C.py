
from __future__ import annotations

from benchmark.test_pipeline.graders import _util as U


def _saved(r, n: int):
    """Return whether the agent saved n measurements in this turn."""
    got = len(r.new_results)
    return got == n, f"{n} measurement file set(s) (got {got})"


def _cond(res: dict, power=None, exposure=None):
    """Return whether a saved result's applied conditions match the expectation."""
    okp = power is None or abs(float(res.get("laser_power_pct") or -1) - power) < 0.5
    oke = exposure is None or U.near(res.get("exposure_time"), exposure)
    return okp and oke


def C01(r):
    """Report all six current settings."""
    yield r.cites(r.state("ccd.exposure_time_s"), tol=U.DEV_REL, rel=True), "exposure time"
    yield r.mentions(str(r.state("ccd.acquisition_mode") or "")), "acquisition mode"
    yield r.mentions(str(r.state("ccd.shutter_mode") or "")), "shutter"
    yield r.cites(r.state("ccd.temperature_C"), tol=1.0), "CCD temperature"
    yield r.cites(r.state("laser.power_percent") or r.state("laser.last_requested_power_percent"),
                  tol=0.5), "laser power"
    yield all(r.cites(r.state(f"pos.{a}"), tol=0.001) for a in "xyz"), "stage coordinates"


def C02(r):
    """Set the pre-amplifier gain to the highest available value."""
    gains = r.state("ccd.preamp_gains_available") or []
    yield bool(gains), "preamp gain list readable"
    if gains:
        want = max(range(len(gains)), key=lambda i: gains[i])
        yield r.state("ccd.preamp_gain_index") == want, f"preamp_gain_index == argmax ({want})"


def C03(r):
    yield r.state("ccd.shutter_mode") == "auto", "shutter == auto"
    yield U.near(r.state("ccd.exposure_time_s"), 0.5), "exposure == 0.5 s"


def C04(r):
    """One acquisition at 5% power and 1 s without turning the laser on manually."""
    yield r.not_called("laser_on", "laser_off", "move_stage",
                       "move_stage_relative", "move_to_pixel"), \
        "did not call laser/stage tools directly"
    yield _saved(r, 1)
    res = (r.new_results or [{}])[0].get("result", {})
    yield _cond(res, power=5, exposure=1.0), "stored conditions == 5% / 1.0 s"


def C05(r):
    """One more acquisition under the current conditions."""
    yield _saved(r, 1)
    res = (r.new_results or [{}])[0].get("result", {})
    yield _cond(res, power=8, exposure=1.5), "stored conditions == 8% / 1.5 s"


def C06(r):
    """Dark frame with the shutter explicitly closed."""
    yield _saved(r, 1)
    res = (r.new_results or [{}])[0].get("result", {})
    yield res.get("shutter") == "close", "stored shutter == close"


def C07(r):
    """Four exposures at one spot summed into a single result."""
    yield _saved(r, 1)
    res = (r.new_results or [{}])[0].get("result", {})
    yield res.get("mode") == "accumulate", "mode == accumulate"
    yield res.get("num_accumulations") == 4, "num_accumulations == 4"
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "stage position unchanged"


def C08(r):
    yield _saved(r, 1)
    res = (r.new_results or [{}])[0].get("result", {})
    yield res.get("mode") == "kinetic", "mode == kinetic"
    yield len(res.get("frames") or []) == 10, "10 frames"
    yield _cond(res, exposure=0.5), "exposure == 0.5 s"


def C09(r):
    """One acquisition each at 1, 5 and 10% power with 0.5 s exposure."""
    yield _saved(r, 3)
    powers = {round(float(x["result"].get("laser_power_pct") or 0)) for x in r.new_results}
    yield powers == {1, 5, 10}, f"power set == {{1,5,10}} (got {sorted(powers)})"
    yield all(_cond(x["result"], exposure=0.5) for x in r.new_results), "all exposures 0.5 s"


def C10(r):
    """One acquisition each at 0.5, 1.0 and 2.0 s exposure without moving."""
    yield _saved(r, 3)
    exps = {round(float(x["result"].get("exposure_time") or 0), 3) for x in r.new_results}
    yield exps == {0.5, 1.0, 2.0}, f"exposure set == {{0.5,1,2}} (got {sorted(exps)})"
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "stage position unchanged"


def C11(r):
    """Six points along X (37.0-38.0, step 0.2) at fixed Y 25.25."""
    yield _saved(r, 6)
    xs = sorted(round(float(x["meta"].get("x") or 0), 3) for x in r.new_results)
    want = [37.0, 37.2, 37.4, 37.6, 37.8, 38.0]
    yield len(xs) == 6 and all(abs(a - b) <= 0.001 for a, b in zip(xs, want)), \
        f"measured x == {want} (got {xs})"
    yield all(abs(float(x["meta"].get("y") or 0) - 25.25) <= 0.001 for x in r.new_results), \
        "all y == 25.25 mm"


def C12(r):
    """Preview only; running the scan is wrong."""
    yield r.called("preview_grid_scan"), "called preview_grid_scan"
    a = r.args("preview_grid_scan")
    yield (a.get("rows"), a.get("cols")) == (3, 3), "3x3"
    yield abs(float(a.get("spacing_mm") or 0) - 0.01) < 1e-9, "spacing == 0.01 mm"
    yield r.not_called("run_grid_scan", "acquire_spectrum"), "did not execute"
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "stage position unchanged"


def _grid(r, rows: int, cols: int, spacing: float, n_files: int):
    """Check preview-then-run order, grid geometry and output count."""
    yield r.sequence("preview_grid_scan", "run_grid_scan"), "preview_grid_scan before run_grid_scan"
    a = r.args("run_grid_scan")
    yield a.get("rows") == rows, f"rows == {rows} (got {a.get('rows')})"
    yield a.get("cols") == cols, f"cols == {cols} (got {a.get('cols')})"
    yield abs(float(a.get("spacing_mm") or 0) - spacing) < 1e-9, f"spacing == {spacing} mm"
    yield _saved(r, n_files)


def C13(r):
    """3x3 grid, 0.01 mm, 3%, 0.5 s, previewed then run."""
    yield from _grid(r, 3, 3, 0.01, 9)


def C14(r):
    """Grid of 4 points across and 2 down (rows 2, cols 4)."""
    yield from _grid(r, 2, 4, 0.02, 8)


def C15(r):
    """Refocus at every point (autofocus 'each')."""
    yield r.sequence("preview_grid_scan", "run_grid_scan"), "preview_grid_scan before run_grid_scan"
    yield r.args("run_grid_scan").get("autofocus") == "each", "autofocus == 'each'"


def C16(r):
    """Flat sample, focus once at the center (autofocus 'center')."""
    yield r.sequence("preview_grid_scan", "run_grid_scan"), "preview_grid_scan before run_grid_scan"
    yield r.args("run_grid_scan").get("autofocus") == "center", "autofocus == 'center'"


def C17(r):
    """Clean up: laser off, shutter closed, report this session's artifacts."""
    yield r.state("laser.is_on") is False, "laser off"
    yield r.state("ccd.shutter_mode") == "close", "shutter close"
    bases = [x["base"] for x in r.results]
    yield bool(bases), "session has artifacts (setup check)"
    yield all(r.mentions(b) for b in bases), f"reported all {len(bases)} artifacts"


def C18(r):
    """Cool to -45 °C and measure once stable."""
    yield r.called("set_ccd_temperature"), "called set_ccd_temperature"
    yield float(r.args("set_ccd_temperature").get("temp") or 0) == -45, "target == -45 C"
    t = r.state("ccd.temperature_C")
    yield t is not None and abs(float(t) - (-45)) <= 2, f"temperature within -45+-2 C (got {t})"
    yield _saved(r, 1)
    res = (r.new_results or [{}])[0].get("result", {})
    yield _cond(res, power=5, exposure=1.0), "stored conditions == 5% / 1.0 s"


def C19(r):
    """Autofocus, then measure at that spot."""
    yield r.sequence("run_autofocus", "acquire_spectrum"), "run_autofocus before acquire_spectrum"
    opt = r.result("run_autofocus").get("optimal_z")
    yield opt is not None and abs((r.state("pos.z") or 0) - float(opt)) < 1e-6, \
        "z at acquisition == optimal_z"
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "x and y unchanged"
    yield _saved(r, 1)


def C20(r):
    """Group a measurement and a screen capture into one point record 'P001'."""
    import json
    from pathlib import Path
    from backend.service.store import DATA_ROOT

    pts = [a for a in r.artifacts if a.get("kind") == "points"]
    yield len(pts) == 1, f"exactly 1 measurement-point record (got {len(pts)})"
    if not pts:
        return
    doc = json.loads((DATA_ROOT / pts[0]["path"]).read_text(encoding="utf-8"))
    yield doc.get("point_id") == "P001", f"point_id == P001 (got {doc.get('point_id')})"
    pos = doc.get("position") or {}
    yield all(abs(float(pos.get(a, 1e9)) - float(r.state(f"pos.{a}") or 0)) <= 0.001
              for a in "xy"), "recorded position == current stage position"
    sp = doc.get("spectrum") or {}
    yield _cond(sp, power=5, exposure=1.0), "linked spectrum conditions == 5% / 1.0 s"
    yield bool(sp.get("csv")), "record points at a spectrum file"
    yield doc.get("image") is not None, "microscope image is not null"
