"""Agent tools for spectrum acquisition and grid scans."""

from __future__ import annotations

import time

from backend.common.config.hw import CAMERA_HEIGHT, CAMERA_WIDTH, STAGE_MAX_X, STAGE_MAX_Y
from backend.common import spectrum_check as _spec_check
from backend.common import vision as _vis
from backend.repo.store.spectrum_store import save_preview_png as _store_save_preview, save_spectrum as _store_save_spectrum
from backend.common import optics_map as _om
from backend.common.result import fail, ok
from backend.service.tools.schema import INTERNAL
from pydantic import Field
from typing import Annotated, Literal, Optional
from backend.repo.hw_access import hw_core as _hw
from backend.service.tools.hw_tools.camera_tools import run_autofocus
from backend.service.memory import recorder as _mem
from backend.service.tools.hw_tools.guards import _ccd_ready, _check_ccd_positive, _serialized, _stage_unavailable
from backend.service.tools.hw_tools.hw_ops import (
    _ACQ_MODES_1D, _READ_MODES_1D, _SHUTTER_MODES, _apply_acq_mode, _apply_laser_power,
    _apply_read_mode, _apply_shutter, _current_read_mode, _effective_shutter,
    _laser_off_quiet, _restore_guide_beam_quiet, mw_to_pct, pct_to_mw,
    POWER_MAX_MW, POWER_MIN_MW, POWER_RANGE_TXT,
)
from backend.service.tools.session_state import _cache_and_return, _sstate
from backend.service.tools.hw_tools.stage_tools import move_stage


def _persist_spectrum(result: dict, tag: str = "") -> dict:
    """Save a measurement result and attach the saved paths and URLs."""
    meta: dict = {}
    if tag:
        meta["tag"] = tag
    try:
        if _hw._stage is not None:
            pos = _hw._stage.get_position()
            meta["x"], meta["y"] = round(float(pos[0]), 3), round(float(pos[1]), 3)
    except Exception:
        pass
    meta.update(_mem.artifact_meta())
    series = [f.get("intensity") or [] for f in result.get("frames") or []] or [result.get("data") or []]
    result["saturated"] = any(_spec_check.saturated(s) for s in series)
    if result["saturated"]:
        result["warning"] = (f"DETECTOR SATURATED: counts reached >= {int(_spec_check.SAT_COUNTS)} "
                             f"(full well {_spec_check.FULL_WELL}). The top of this spectrum is clipped, so peak "
                             f"heights, ratios and SNR from it are not valid. Lower the power or the exposure and "
                             f"measure again before drawing conclusions.")
    try:
        axis, ymean = _mem._spectrum(result)
        if axis is not None:
            from backend.common import raman_judge as _judge
            result["spectrum_qc"] = _judge.spectrum_qc(axis, ymean)
            ctx = _mem.sample_ctx()
            bands = list(ctx.get("expected_bands") or []) + list(ctx.get("marker_bands") or [])
            if bands:
                result["band_check"] = {"sample_name": ctx.get("sample_name"),
                                        "bands": _judge.judge_bands(axis, ymean, bands)}
    except Exception as e:  # noqa: BLE001
        result["spectrum_qc_error"] = f"{type(e).__name__}: {e}"
    saved = _store_save_spectrum(result, meta or None)
    if saved.get("ok"):
        result["saved"] = saved
    else:
        result["saved_error"] = saved.get("error")
    return result


def _conditions(power_mw, exposure_s, accumulations: int = 1, frames: int | None = None) -> dict:
    """Return the measurement conditions as one line and the total laser exposure time (s)."""
    acc = int(max(accumulations or 1, 1))
    pw = f"{power_mw:g} mW" if power_mw is not None else "power unknown"
    text = f"{pw}, {float(exposure_s or 0):g} s" + (f" x {acc} accumulations" if acc > 1 else "")
    if frames:
        text += f", {int(frames)} kinetic frames"
    t = float(exposure_s or 0) * acc * int(frames or 1)
    return {"conditions": text, "total_exposure_s": round(t, 3)}


@_serialized("acquire_spectrum")
@_mem.recorded
def acquire_spectrum(
    exposure: Annotated[Optional[float], Field(description="CCD exposure time (seconds). Omit to keep the CCD's current exposure (e.g. one set earlier by set_ccd_exposure).")] = None,
    power: Annotated[Optional[float], Field(ge=POWER_MIN_MW, le=POWER_MAX_MW, description=f"Laser power, a real value in {POWER_RANGE_TXT}. Omit ONLY to reuse a power you already set earlier in this session - if none has ever been set the call is REFUSED rather than defaulted, because choosing the laser power and exposure time for an unknown sample is your decision, not the tool's. Higher power gives more signal but photobleaches or burns fragile samples.")] = None,
    stabilize_sec: Annotated[Optional[float], Field(description='Wait time for power stabilization after laser ON (seconds). Default 0.5')] = 0.5,
    acq_mode: Annotated[Optional[Literal['single', 'accumulate', 'kinetic']], Field(description="CCD acquisition mode. Omit to keep the CCD's current mode. 'single': single shot. 'accumulate': num_accumulations shots back to back at the minimum interval; the returned spectrum is their per-shot AVERAGE. 'kinetic': acquire kinetic_count frames continuously -> time-series analysis.")] = None,
    num_accumulations: Annotated[Optional[int], Field(ge=1, description='Accumulations per frame in accumulate/kinetic mode. Omit to keep the current value.')] = None,
    kinetic_count: Annotated[Optional[int], Field(ge=1, description='Total number of frames to acquire in kinetic mode. Omit to keep the current value.')] = None,
    kinetic_cycle_time: Annotated[Optional[float], Field(description='Frame interval in kinetic mode (seconds). If omitted, the SDK auto-computes the minimum.')] = None,
    read_mode: Annotated[Optional[Literal['fvb', 'single_track']], Field(description="CCD readout mode. Omit to keep the CCD's current read mode. 'fvb': Full Vertical Binning - sum all rows, 1D spectrum. 'single_track': read only a specific track - single_track_center required.")] = None,
    hbin: Annotated[Optional[int], Field(ge=1, description='Horizontal binning pixel count. Omit to keep the current value.')] = None,
    single_track_center: Annotated[Optional[int], Field(description="Center pixel row number when read_mode='single_track'. Omit to reuse the currently configured track.")] = None,
    single_track_width: Annotated[Optional[int], Field(ge=1, description="Track width (pixels) when read_mode='single_track'. Omit to keep the current value.")] = None,
    shutter: Annotated[Optional[Literal['auto', 'open', 'close']], Field(description="Shutter mode for this acquisition. Omit to KEEP the current setting - including one you set earlier with set_ccd_shutter. If nobody has ever set it, it opens to 'auto' (the CCD boots closed, so keeping that would silently hand you a dark frame). Use 'close' to acquire a DARK / background frame with no light reaching the detector - that is the supported way to measure a dark reference.")] = None,
    restore_guide_beam: Annotated[Optional[bool], Field(json_schema_extra=INTERNAL, description='Internal: restore the guide beam after acquisition. Consecutive callers (run_grid_scan) pass False to avoid toggling it on every point.')] = True,
    purpose: Annotated[Optional[str], Field(description="One sentence: why THIS shot, and why these power/exposure/frames - e.g. 'pre-irradiation low-power probe of spot B2; 0.5 mW x 2 s is 10x below the lowest power that changed this sample'. Saved in the measurement record so the choice can be audited later. Always give it.")] = None,
    spot_label: Annotated[Optional[str], Field(description="Your own short name for the spot being measured (e.g. 'A1', 'bareSi-1', 'cluster3'). Use the same label every time you return to that spot, so shots on one spot can be grouped even if the stage position drifts slightly.")] = None,
) -> dict:
    """Acquire a Raman spectrum in single, accumulate or kinetic mode.

    Sets the laser power, turns the laser on, acquires, turns the laser off and returns the optics to the guide-beam position. The laser is turned off even if the acquisition fails. Omitted arguments keep the current instrument settings.

    Returns:
        dict: ok() with intensities, Raman shift axis, saturation flag, the exposure and power read back from the instrument, and the saved file, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    if power is None:
        last = pct_to_mw(getattr(_hw._laser, "power_pct", None))
        if last is None:
            return fail("No laser power has been set in this session, so there is nothing to 'keep' and this "
                        "tool will not pick one for you - firing an unknown sample at an arbitrary power can "
                        "photobleach or burn it. Decide the power yourself and pass acquire_spectrum(power=...), "
                        "or set it once with set_laser_power(mw) and then measure. If you do not know the "
                        "sample's tolerance, start low (well under 1 mW), look at the signal, and raise it.")
        eff_power = float(last)
    else:
        eff_power = power

    eff_acq = acq_mode or getattr(_hw._ccd, 'aq_mode', None) or 'single'
    if eff_acq not in _ACQ_MODES_1D:
        return fail(f"acquire_spectrum supports {list(_ACQ_MODES_1D)}, but the CCD is "
                    f"currently in '{eff_acq}' mode. Pass acq_mode explicitly, or call "
                    f"set_ccd_acquisition_mode first.")

    eff_read = read_mode or _current_read_mode()
    if eff_read not in _READ_MODES_1D:
        return fail(f"acquire_spectrum assembles a 1D spectrum, but the CCD read mode is '{eff_read}'. "
                    f"Pass read_mode='fvb' (or 'single_track'), or call set_ccd_read_mode first.")

    eff_center = (single_track_center if single_track_center is not None
                  else getattr(_hw._ccd, 'ro_single_track_center', None))
    if eff_read == 'single_track' and eff_center is None:
        return fail("single_track_center is required when read_mode='single_track'")

    for _n, _v, _i in (("exposure", exposure, False),
                       ("num_accumulations", num_accumulations, True),
                       ("kinetic_count", kinetic_count, True),
                       ("kinetic_cycle_time", kinetic_cycle_time, False),
                       ("hbin", hbin, True),
                       ("single_track_width", single_track_width, True),
                       ("single_track_center", eff_center if eff_read == 'single_track' else None, True)):
        err = _check_ccd_positive(_n, _v, integer=_i)
        if err:
            return err

    eff_shutter = _effective_shutter(shutter)
    if eff_shutter not in _SHUTTER_MODES:
        return fail(f"shutter must be one of {list(_SHUTTER_MODES)}.")

    eff_trigger = 'internal'

    _mem.before_fire(purpose=purpose, spot_label=spot_label)

    err = _apply_laser_power(eff_power, settle_s=stabilize_sec)
    if err:
        return err

    raw = None
    try:
        _hw._laser.laser_on()
        _mem.fired(pct_to_mw(getattr(_hw._laser, "power_pct", None)))
        time.sleep(stabilize_sec)

        err = _apply_acq_mode(eff_acq, exposure=exposure,
                              num_accumulations=num_accumulations,
                              kinetic_count=kinetic_count,
                              kinetic_cycle_time=kinetic_cycle_time)
        if err:
            return err

        err = _apply_read_mode(eff_read, hbin=hbin, single_track_center=eff_center,
                               single_track_width=single_track_width)
        if err:
            return err

        if getattr(_hw._ccd, 'trigger_mode', None) != 'internal':
            _hw._ccd.set_trigger_mode('internal')

        err = _apply_shutter(eff_shutter, explicit=(shutter is not None))
        if err:
            return err

        _hw._ccd.free_internal_memory()

        try:
            eff_exposure = float(_hw._ccd.get_exposure_time())
        except Exception:
            eff_exposure = float(exposure) if exposure is not None else 0.0
        eff_num_acc = int(getattr(_hw._ccd, 'num_acc', 1) or 1)
        eff_num_kin = int(getattr(_hw._ccd, 'num_kin', 1) or 1)

        if eff_acq == 'kinetic':
            _hw._ccd.prepare_acquisition()
            _hw._ccd.start_acquisition()

            _cyc = max(float(getattr(_hw._ccd, 'kinetic_cycle_time', 0) or 0), eff_exposure + 0.1)
            _timeout_s = _cyc * max(eff_num_kin, 1) * max(eff_num_acc, 1) * 2 + 15.0
            _deadline = time.time() + _timeout_s
            while _hw._ccd.get_status() != 'IDLE':
                if time.time() > _deadline:
                    try:
                        _hw._ccd.abort_acquisition()
                    except Exception:
                        pass
                    raise TimeoutError(
                        f"kinetic acquisition timeout: exceeded {_timeout_s:.1f} s "
                        f"(trigger={eff_trigger}, frames={eff_num_kin})"
                    )
                time.sleep(0.05)
            raw = _hw._ccd.get_acquired_data()
        else:
            _cycle = max(eff_exposure, float(getattr(_hw._ccd, 'accumulation_time', 0) or 0))
            _timeout_ms = int((_cycle * max(eff_num_acc, 1) * 2 + 15) * 1000)
            raw = _hw._ccd.start_acquisition_cycle(
                trigger_mode_str=eff_trigger,
                timeout_ms=_timeout_ms,
            )

    except Exception as e:
        return fail(str(e))

    finally:
        _laser_off_quiet()
        try:
            if _hw._ccd.get_status() != 'IDLE':
                _hw._ccd.abort_acquisition()
        except Exception:
            pass
        if restore_guide_beam:
            _restore_guide_beam_quiet()
            _mem.after_restore()

    if raw is None:
        return fail("CCD data acquisition failed")

    if eff_acq == 'kinetic':
        cal = getattr(_hw._ccd, '_calibrator', None)
        frames = []
        for i in range(raw.shape[0]):
            frame_intensity = raw[i].flatten().tolist()
            frame = {
                "frame_index": i,
                "intensity": frame_intensity,
                "length": len(frame_intensity),
                "max_intensity": float(max(frame_intensity)),
                "sum_intensity": float(sum(frame_intensity)),
                "calibrated": False,
            }
            if cal is not None:
                pixels = range(len(frame_intensity))
                frame.update({
                    "calibrated": True,
                    "raman_shift_cm-1": [float(cal.pixel_to_raman_shift(p)) for p in pixels],
                    "wavelength_nm":    [float(cal.pixel_to_wavelength(p))   for p in pixels],
                    "laser_nm":         float(cal.laser_nm),
                })
            frames.append(frame)
        return _persist_spectrum(ok(mode="kinetic",
                                    num_frames=len(frames),
                                    kinetic_count=eff_num_kin,
                                    num_accumulations=eff_num_acc,
                                    exposure_time=eff_exposure,
                                    laser_power_mw=eff_power,
                                    laser_power_pct=round(mw_to_pct(eff_power), 6),
                                    trigger_mode=eff_trigger,
                                    shutter=eff_shutter,
                                    frames=frames,
                                    **_conditions(eff_power, eff_exposure, eff_num_acc, frames=eff_num_kin)))
    else:
        data = raw
        if data.get("error"):
            return fail(data["error"])
        intensity = data["intensity"]
        if eff_acq == 'accumulate' and eff_num_acc > 1:
            intensity = [v / eff_num_acc for v in intensity]
        result = ok(mode=eff_acq,
                    length=len(intensity),
                    max_intensity=float(max(intensity)) if intensity else 0.0,
                    sum_intensity=float(sum(intensity)) if intensity else 0.0,
                    data=intensity,
                    calibrated=data.get("calibrated", False),
                    exposure_time=eff_exposure,
                    laser_power_mw=eff_power,
                    laser_power_pct=round(mw_to_pct(eff_power), 6),
                    trigger_mode=eff_trigger,
                    shutter=eff_shutter)
        if eff_acq == 'accumulate':
            result["num_accumulations"] = eff_num_acc
        result.update(_conditions(eff_power, eff_exposure, eff_num_acc if eff_acq == 'accumulate' else 1))
        if data.get("calibrated"):
            result["raman_shift_cm-1"] = data["raman_shift_cm-1"]
            result["wavelength_nm"]    = data["wavelength_nm"]
            result["laser_nm"]         = data["laser_nm"]
        return _persist_spectrum(result)


_GRID_SPOT_RADIUS_PX = 14  # radius of the grid markers on the preview

_GRID_AF_ABORT_STREAK = 3  # consecutive autofocus failures that stop a scan

_GRID_MAX_POINTS = 400  # maximum points per grid


def _gate() -> dict:
    """Return the grid approval gate state of this session."""
    return _sstate()["grid_gate"]


def _grid_gate_geom(rows, cols, spacing_mm, center_x, center_y) -> dict:
    """Normalize the grid geometry used to compare preview and run."""
    return {
        "rows": int(rows), "cols": int(cols),
        "spacing_mm": round(float(spacing_mm), 4),
        "center_x": None if center_x is None else round(float(center_x), 4),
        "center_y": None if center_y is None else round(float(center_y), 4),
    }


def grid_gate_begin_turn(interactive: bool = True) -> None:
    """Update the grid-scan approval gate at the start of a user turn.

    Args:
        interactive (bool): True for chat sessions, where a preview must be approved before a scan. False for unattended runs.
    """
    _gate()["enforce"] = bool(interactive)
    if not interactive:
        _gate()["geom"] = None
        _gate()["resolved"] = None
        _gate()["state"] = "none"
        return
    if _gate()["state"] == "pending":
        _gate()["state"] = "armed"
    elif _gate()["state"] == "armed":
        _gate()["state"] = "none"
        _gate()["geom"] = None
        _gate()["resolved"] = None


def _grid_gate_on_preview(geom: dict, resolved: tuple) -> None:
    """Put the gate into the pending state after a successful preview."""
    _gate()["geom"] = geom
    _gate()["resolved"] = {"x": round(float(resolved[0]), 4),
                           "y": round(float(resolved[1]), 4)}
    _gate()["state"] = "pending"


def _grid_gate_check(geom: dict):
    """Check grid approval before a run; return an error dict or None."""
    if not _gate()["enforce"]:
        return None
    state = _gate()["state"]
    if state == "none" or _gate()["geom"] is None:
        return fail("Human approval required: no approved grid preview is on record. Call preview_grid_scan "
                    "first, show the user the preview, end your turn, and run only after the user approves.")
    if state == "pending":
        return fail("Human approval required: the grid preview has NOT been approved yet. Do not run in the "
                    "same turn as the preview - end your turn now, let the user see the preview, and call "
                    "run_grid_scan only after the user explicitly approves it in a new message.")
    if geom != _gate()["geom"]:
        return fail(f"Approval mismatch: the user approved grid {_gate()['geom']} but this run requests "
                    f"{geom}. Preview the exact grid again and get the user's approval before running.")
    return None


def _grid_gate_approved_center():
    """Return the grid center of the approved preview, or None."""
    if not _gate()["enforce"] or _gate()["state"] != "armed":
        return None
    r = _gate().get("resolved")
    if not r:
        return None
    return (float(r["x"]), float(r["y"]))


def _grid_gate_consume() -> None:
    """Consume one approval just before firing."""
    if not _gate()["enforce"]:
        return
    _gate()["geom"] = None
    _gate()["resolved"] = None
    _gate()["state"] = "none"


def _grid_stage_coords(center_x: float, center_y: float,
                       rows: int, cols: int, spacing_mm: float) -> list:
    """Return raster-ordered stage coordinates of a rows × cols grid."""
    pts = []
    idx = 0
    for i in range(rows):
        dy = (i - (rows - 1) / 2.0) * spacing_mm
        for j in range(cols):
            dx = (j - (cols - 1) / 2.0) * spacing_mm
            pts.append((idx, i, j, round(center_x + dx, 4), round(center_y + dy, 4)))
            idx += 1
    return pts


def _validate_grid_args(rows, cols, spacing_mm):
    """Validate common grid arguments; return an error dict or None."""
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        return fail("rows and cols must be integers >= 1.")
    if rows * cols > _GRID_MAX_POINTS:
        return fail(f"Too many points ({rows*cols}); max {_GRID_MAX_POINTS}.")
    try:
        if float(spacing_mm) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return fail("spacing_mm must be a number > 0.")
    return None


def _validate_grid_range(pts):
    """Check that all grid points are within the stage travel range."""
    for idx, i, j, sx, sy in pts:
        if not (0 <= sx <= STAGE_MAX_X and 0 <= sy <= STAGE_MAX_Y):
            return fail(f"Point {idx} (row {i}, col {j}) at X={sx}, Y={sy} mm is outside the stage range "
                        f"(0..{STAGE_MAX_X} x 0..{STAGE_MAX_Y}). Adjust center/spacing/size.")
    return None


@_serialized("preview_grid_scan")
def preview_grid_scan(
    rows: Annotated[int, Field(description='Number of grid points stacked VERTICALLY = grid HEIGHT (stage Y axis), integer >= 1. e.g. rows=3 -> 3 points tall.')],
    cols: Annotated[int, Field(description='Number of grid points side-by-side HORIZONTALLY = grid WIDTH (stage X axis), integer >= 1. e.g. cols=2 -> 2 points wide.')],
    spacing_mm: Annotated[float, Field(description='Distance between adjacent points (mm), > 0')],
    center_x: Annotated[Optional[float], Field(description='Grid center X (mm). Optional; defaults to current stage X')] = None,
    center_y: Annotated[Optional[float], Field(description='Grid center Y (mm). Optional; defaults to current stage Y')] = None,
) -> dict:
    """Draw the grid positions on the current camera image without moving or firing.

    Returns:
        dict: ok() with the overlay image and the number of points inside the field of view, or fail().
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    try:
        import cv2

        spacing_mm = float(spacing_mm)
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")
        cur_x, cur_y = float(pos[0]), float(pos[1])
        cx = cur_x if center_x is None else float(center_x)
        cy = cur_y if center_y is None else float(center_y)

        frame = _hw._camera.get_latest_frame()
        if frame is None:
            return fail("Failed to acquire frame (check whether streaming is active)")
        frame_bgr = _vis.to_view_bgr(frame)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        range_err = _validate_grid_range(pts)
        if range_err:
            return range_err
        n_in_view = 0
        for idx, i, j, sx, sy in pts:
            px, py = _om.stage_to_pixel(sx, sy, cur_x, cur_y)
            ipx, ipy = int(round(px)), int(round(py))
            if 0 <= ipx < CAMERA_WIDTH and 0 <= ipy < CAMERA_HEIGHT:
                n_in_view += 1
            cv2.circle(frame_bgr, (ipx, ipy), _GRID_SPOT_RADIUS_PX, (0, 255, 0), 2)

        ret, buf = cv2.imencode('.png', frame_bgr)
        if not ret:
            return fail("PNG encoding failed")
        img_b64 = _vis.encode_for_model(frame_bgr)
        if img_b64 is None:
            return fail("JPEG encoding failed")
        saved_img = _store_save_preview(buf.tobytes(), tag="grid_preview")

        fov_x, fov_y = _om.fov_mm()
        span_x, span_y = (cols - 1) * spacing_mm, (rows - 1) * spacing_mm
        question = (
            f"Grid scan PREVIEW (not executed yet): {rows} rows x {cols} cols = {rows*cols} points "
            f"-> a {cols}-wide x {rows}-tall grid, {spacing_mm} mm spacing, "
            f"centered at stage (X={cx:.4f}, Y={cy:.4f}) mm. "
            f"Green circles mark the points to be scanned. "
            f"Grid span {span_x:.3f}x{span_y:.3f} mm vs camera view ~{fov_x:.3f}x{fov_y:.3f} mm; "
            f"{n_in_view}/{rows*cols} points fall within the current view (the rest are outside the frame "
            f"but will still be measured). "
            f"STOP HERE - do NOT run the scan yet. Show this preview to the user, confirm the "
            f"{cols}-wide x {rows}-tall orientation is what they asked for, then END YOUR TURN and WAIT "
            f"for their explicit approval. Only in a LATER turn, after the user approves, call run_grid_scan "
            f"with these SAME parameters. If the layout is wrong, call preview_grid_scan again with adjusted "
            f"rows/cols/spacing_mm/center."
        )
        out = ok(rows=rows,
                 cols=cols,
                 spacing_mm=spacing_mm,
                 center={"x": round(cx, 4), "y": round(cy, 4)},
                 n_points=rows * cols,
                 n_in_view=n_in_view,
                 fov_mm={"x": round(fov_x, 4), "y": round(fov_y, 4)},
                 image_base64=img_b64,
                 question=question)
        if saved_img.get("ok"):
            out["image_file"] = saved_img["file_id"]
            out["saved"] = {"title": f"Grid preview {rows}x{cols} @ {spacing_mm}mm",
                            "image_url": saved_img["image_url"]}
        _grid_gate_on_preview(_grid_gate_geom(rows, cols, spacing_mm, center_x, center_y),
                              (cx, cy))
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("run_grid_scan")
def run_grid_scan(
    rows: Annotated[int, Field(description='Number of grid points stacked VERTICALLY = grid HEIGHT (stage Y axis), integer >= 1. e.g. rows=3 -> 3 points tall. Must match the approved preview.')],
    cols: Annotated[int, Field(description='Number of grid points side-by-side HORIZONTALLY = grid WIDTH (stage X axis), integer >= 1. e.g. cols=2 -> 2 points wide. Must match the approved preview.')],
    spacing_mm: Annotated[float, Field(description='Distance between adjacent points (mm), > 0')],
    exposure: Annotated[float, Field(description='Exposure time per point (s) - REQUIRED. Choose it for THIS sample rather than reusing a number: too short buries the peaks in read noise, too long saturates the detector and multiplies the total run time by the point count.')],
    power: Annotated[float, Field(description=f"Laser power per point, {POWER_RANGE_TXT} - REQUIRED. It is applied at EVERY point, so the sample sees it rows*cols times. Higher power raises signal but photobleaches or burns fragile samples (biological, polymer, thin film); if you are unsure of the sample's tolerance, start low and check one point before committing the whole grid.")],
    autofocus: Annotated[Literal['each', 'center', 'none'], Field(description="Autofocus strategy. 'each' = autofocus at every point (most accurate, slowest); 'center' = autofocus once at the grid center then reuse that Z (fast, for flat samples); 'none' = no autofocus, keep current Z. REQUIRED - this is a real trade-off, not a formality: 'each' costs an extra Z sweep (and guide-beam exposure) at every point, which on a large grid dominates the run time, while 'center' or 'none' will drift out of focus on a tilted or uneven sample and quietly return weak spectra. Decide from what you know about the sample's flatness.")],
    center_x: Annotated[Optional[float], Field(description='Grid center X (mm). Optional; defaults to current stage X')] = None,
    center_y: Annotated[Optional[float], Field(description='Grid center Y (mm). Optional; defaults to current stage Y')] = None,
) -> dict:
    """Run a grid scan.

    At each point the stage moves, autofocus runs if requested, and a spectrum is acquired and saved.

    Returns:
        dict: ok() with a summary of the scan, or fail().
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    if autofocus not in ("each", "center", "none"):
        return fail("autofocus must be one of: each, center, none.")
    gate_err = _grid_gate_check(_grid_gate_geom(rows, cols, spacing_mm, center_x, center_y))
    if gate_err:
        return gate_err
    approved_center = _grid_gate_approved_center()
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    if _hw._ccd is None:
        return fail("CCD is not initialized (cooling or not connected).")
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    if autofocus != "none" and _hw._camera is None:
        return fail("Camera is not initialized (required for autofocus). "
                    "Use autofocus='none' to skip.")

    spacing_mm = float(spacing_mm)
    exposure, power = float(exposure), float(power)
    n = rows * cols

    fired = False

    try:
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")
        fallback_x, fallback_y = (approved_center if approved_center
                                  else (float(pos[0]), float(pos[1])))
        cx = fallback_x if center_x is None else float(center_x)
        cy = fallback_y if center_y is None else float(center_y)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        range_err = _validate_grid_range(pts)
        if range_err:
            return range_err

        if autofocus == "center":
            mv = move_stage(x=cx, y=cy)
            if not mv.get("ok"):
                return fail(f"Failed to move to grid center: {mv.get('error')}")
            af = run_autofocus()
            if not af.get("ok"):
                return fail(f"Autofocus at grid center failed: {af.get('error')}")

        _grid_gate_consume()

        results = []
        n_ok = 0
        n_frames = None
        af_failed, af_limit_hits, af_streak = 0, 0, 0
        aborted = None
        for idx, i, j, sx, sy in pts:
            mv = move_stage(x=sx, y=sy)
            if not mv.get("ok"):
                results.append(fail(mv.get("error"), i=idx, row=i, col=j, x=sx, y=sy))
                continue

            af_ok = None
            if autofocus == "each":
                af = run_autofocus()
                af_ok = bool(af.get("ok"))
                if af_ok:
                    af_streak = 0
                    if af.get("z_limit_hits"):
                        af_limit_hits += 1
                else:
                    af_failed += 1
                    af_streak += 1
                    if af_streak >= _GRID_AF_ABORT_STREAK:
                        aborted = {
                            "at_point": idx, "row": i, "col": j,
                            "reason": (
                                f"Autofocus failed {af_streak} times in a row (last error: "
                                f"{af.get('error')}). That is a systematic problem, not a bad point - "
                                f"the camera may have stopped streaming, the guide beam may be off, or "
                                f"the sample may sit outside the Z travel range. The scan was STOPPED "
                                f"here instead of firing the laser at the remaining points out of focus."),
                        }
                        break

            fired = True
            res = _cache_and_return(acquire_spectrum(exposure=exposure, power=power,
                                                     restore_guide_beam=False))
            if res.get("ok"):
                n_ok += 1
                if n_frames is None:
                    n_frames = _mem._applied(res)["n_frames"]
                files = (res.get("saved") or {}).get("files") or {}
                ref = files.get("csv") or files.get("png") or ""
                fname = ref.replace("\\", "/").rsplit("/", 1)[-1]
                rec = {"i": idx, "x": sx, "y": sy,
                       "max_intensity": res.get("max_intensity"), "file": fname}
                if af_ok is False:
                    rec["autofocus_failed"] = True
                results.append(rec)
            else:
                results.append(fail(res.get("error"), i=idx, row=i, col=j, x=sx, y=sy))

        oks = [r for r in results if "max_intensity" in r]
        fails = [r for r in results if r.get("ok") is False]
        inten = [r["max_intensity"] for r in oks if r.get("max_intensity") is not None]
        out = {
            "ok": aborted is None,
            "rows": rows, "cols": cols, "spacing_mm": spacing_mm,
            "center": {"x": round(cx, 4), "y": round(cy, 4)},
            "autofocus": autofocus, "exposure": exposure, "power": power,
            "n_frames": n_frames,
            "n_points": n, "n_measured": n_ok, "n_failed": len(fails),
            "intensity": ({"min": min(inten), "max": max(inten),
                           "mean": round(sum(inten) / len(inten), 1)} if inten else None),
            "note": ("Each point was auto-saved with its (x,y) tag. Use aggregate_spectra_csv / "
                     "combine_spectra / bundle_results / run_analysis to merge or inspect per-point data."),
        }
        if aborted is not None:
            out["aborted"] = aborted
            out["error"] = aborted["reason"]
            out["note"] = (
                f"STOPPED EARLY at point {aborted['at_point']} of {n}. The {n_ok} point(s) measured "
                f"before that were auto-saved and are still usable, but the grid is INCOMPLETE - do not "
                f"report it as finished. Fix the focus problem (check the camera stream and the sample "
                f"height with run_autofocus on its own), then preview and re-run the grid.")
        if autofocus == "each" and af_failed:
            out["n_autofocus_failed"] = af_failed
            out.setdefault("warnings", []).append(
                f"Autofocus failed at {af_failed} of the points that were measured. Those spectra were "
                f"taken at whatever Z the stage happened to be at, so they may be out of focus and their "
                f"intensities are not comparable with the rest - the affected points are marked with "
                f"autofocus_failed. Do not read a weak signal there as a property of the sample.")
        if af_limit_hits:
            out["n_autofocus_z_limit"] = af_limit_hits
            out.setdefault("warnings", []).append(
                f"At {af_limit_hits} point(s) the focus search ran into the Z travel limit, meaning the "
                f"true focus is probably outside the reachable range. Repeating the scan will not help - "
                f"the sample or the objective needs to be repositioned.")
        if fails:
            out["failed_points"] = fails[:10]
        if n <= 32:
            out["points"] = oks
        return out
    except Exception as e:
        return fail(str(e))
    finally:
        if fired:
            _restore_guide_beam_quiet()
