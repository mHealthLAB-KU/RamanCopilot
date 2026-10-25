"""Agent tools for the microscope camera, autofocus and pixel-based stage moves."""

from __future__ import annotations

import time

from backend.common.config.hw import CAMERA_HEIGHT, CAMERA_WIDTH, STAGE_MAX_Z, STAGE_MIN_Z
from backend.common import vision as _vis
from backend.repo.store.spectrum_store import save_preview_png as _store_save_preview, save_scene as _store_save_scene
from backend.common import optics_map as _om
from backend.common.result import fail, ok
from pydantic import Field
from typing import Annotated, Optional
from backend.repo.hw_access import hw_core as _hw
from backend.service.tools.hw_tools.guards import _check_stage_target, _serialized, _stage_unavailable
from backend.service.tools.hw_tools.hw_ops import _laser_off_quiet
from backend.service.tools.hw_tools.laser_aim import locate_laser_aim
from backend.service.tools.hw_tools.stage_tools import move_stage


@_serialized("start_camera_stream")
def start_camera_stream() -> dict:
    """Start camera streaming.

    Returns:
        dict: ok() with already_streaming, or fail().
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")

    try:
        if getattr(_hw._camera, 'is_streaming', False):
            return ok(already_streaming=True, status="Camera is already streaming.")

        _hw._camera.start_stream()
        return ok(already_streaming=False, status="Camera streaming started successfully.")

    except Exception as e:
        return fail(f"Failed to start streaming: {str(e)}")

@_serialized("stop_camera_stream")
def stop_camera_stream() -> dict:
    """Stop camera streaming.

    Returns:
        dict: ok() with a status message, or fail().
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")

    try:
        if not getattr(_hw._camera, 'is_streaming', False):
            return ok(status="Camera is not currently streaming.")

        _hw._camera.stop_stream()
        return ok(status="Camera streaming stopped successfully.")

    except Exception as e:
        return fail(f"Failed to stop streaming: {str(e)}")


@_serialized("set_camera_exposure")
def set_camera_exposure(
    ms: Annotated[float, Field(description='Exposure time [ms]. e.g. 10.0, 50.0, 100.0')],
) -> dict:
    """Set the camera exposure time.

    Returns:
        dict: ok() with exposure_ms, or fail().
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    if ms <= 0:
        return fail("Exposure time must be greater than 0.")
    try:
        _hw._camera.set_exposure(ms)
        return ok(exposure_ms=ms)
    except Exception as e:
        return fail(str(e))


@_serialized("set_camera_auto_exposure")
def set_camera_auto_exposure(
    enabled: Annotated[bool, Field(description='true = auto exposure ON, false = manual exposure')],
) -> dict:
    """Enable or disable camera auto exposure.

    Returns:
        dict: ok() with auto_exposure, or fail().
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    try:
        _hw._camera.set_auto_exposure(bool(enabled))
        return ok(auto_exposure=enabled)
    except Exception as e:
        return fail(str(e))


_AF_UNMEASURED_ABORT = 3  # consecutive unmeasured frames that stop autofocus

FOCUS_RECHECK_RANGE = (0.7, 1.4)  # allowed ratio of the rechecked area to the best area
FOCUS_BRACKET_FACTOR = 1.5  # both sides of the best Z must exceed the best area by this factor
FOCUS_AREA_WARN_PX = 11000  # best areas above this are reported as uncertain


def _focus_check(z_scores: list, best_area: float, optimal_z: float, recheck_area) -> dict:
    """Judge the quality of an autofocus result (ok, uncertain or failed)."""
    reasons, verdict = [], "ok"
    zs = [(float(d["z"]), float(d["area_px"])) for d in z_scores if d.get("area_px") is not None]
    lo = [a for z, a in zs if z < optimal_z - 1e-6]
    hi = [a for z, a in zs if z > optimal_z + 1e-6]
    ratio = (float(recheck_area) / best_area) if recheck_area and best_area else None
    if ratio is None:
        verdict = "failed"
        reasons.append("the spot could not be re-measured at the chosen Z")
    elif not FOCUS_RECHECK_RANGE[0] <= ratio <= FOCUS_RECHECK_RANGE[1]:
        verdict = "failed"
        reasons.append(f"re-measured spot area at the chosen Z is {ratio:.2f}x the best area - the focus "
                       f"metric is not reproducible (normal: 0.86-1.07x)")
    if not (lo and max(lo) >= FOCUS_BRACKET_FACTOR * best_area and hi and max(hi) >= FOCUS_BRACKET_FACTOR * best_area):
        verdict = "failed"
        reasons.append("no clear minimum: the best Z is not bracketed on both sides by clearly larger spots, "
                       "so the true focus may lie outside the searched range")
    if best_area > FOCUS_AREA_WARN_PX and verdict == "ok":
        verdict = "uncertain"
    if best_area > FOCUS_AREA_WARN_PX:
        reasons.append(f"best spot area {int(best_area)} px is > {FOCUS_AREA_WARN_PX} px (2.5x the typical "
                       f"in-focus 4461 px on this instrument) - the spot is still spread or the surface "
                       f"scatters/fluoresces strongly")
    return {"verdict": verdict, "recheck_ratio": None if ratio is None else round(ratio, 3),
            "reasons": reasons}

@_serialized("run_autofocus")
def run_autofocus(
    initial_z: Annotated[Optional[float], Field(description='Starting Z position for the search (mm). If omitted, keep the current Z')] = None,
    step_size: Annotated[Optional[float], Field(description='Initial Z step size (mm). Default 0.030 (30 um)')] = 0.03,
    min_step: Annotated[Optional[float], Field(description='Minimum step size (mm) - the search ends below this. Default 0.001 (1 um)')] = 0.001,
    max_steps: Annotated[Optional[int], Field(description='Maximum number of steps - forced stop when exceeded. Default 100')] = 100,
) -> dict:
    """Autofocus by minimizing the guide-beam spot area.

    Hill climbing in Z: the step direction reverses and the step size halves when the area increases. After convergence the laser spot position is measured and stored.

    Returns:
        dict: ok() with optimal_z and best_area, or fail().
    """
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    if not getattr(_hw._stage, "has_z", True):
        return fail("This stage has no Z axis, so autofocus cannot run. The focus is fixed; "
                    "proceed with the measurement without focusing.")
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    if initial_z is not None:
        err = _check_stage_target(z=initial_z)
        if err:
            return err

    _bad = {}
    if step_size is None or float(step_size) <= 0:
        _bad["step_size"] = step_size
    if min_step is None or float(min_step) <= 0:
        _bad["min_step"] = min_step
    if max_steps is None or int(max_steps) < 1:
        _bad["max_steps"] = max_steps
    if _bad:
        return fail(
            f"Rejected: {', '.join(f'{k}={v}' for k, v in _bad.items())}. "
            f"Autofocus searches by moving Z in steps; a step of zero (or a step floor of "
            f"zero, or no step budget) means there is no search at all, and the tool would "
            f"report success without ever focusing. Nothing was changed. Omit these "
            f"parameters to use the defaults (step_size=0.03, min_step=0.001, max_steps=100).",
            rejected=_bad)
    if float(step_size) < float(min_step):
        return fail(
            f"Rejected: step_size={step_size} is smaller than min_step={min_step}, so the "
            f"search would end before it began and the focus would be left unchanged. "
            f"min_step is the point at which the search stops refining, so it must be "
            f"smaller than the starting step.",
            step_size=step_size, min_step=min_step)

    try:
        pos = _hw._stage.get_position()
        cur_x, cur_y, cur_z = pos[0], pos[1], pos[2]
        cur_a = pos[3] if len(pos) > 3 else 0
        n_clamped = 0

        def _goto_z(z: float) -> float:
            """Move Z, clipped to the allowed range."""
            nonlocal n_clamped
            zc = max(STAGE_MIN_Z, min(STAGE_MAX_Z, float(z)))
            if zc != float(z):
                n_clamped += 1
            _hw._stage.move_absolute(cur_x, cur_y, zc, cur_a)
            time.sleep(0.3)
            return zc

        if initial_z is not None:
            cur_z = _goto_z(initial_z)

        _hw._laser.set_guide_beam()
        if not getattr(_hw._camera, 'is_streaming', False):
            _hw._camera.start_stream()

        def _capture_spot_area():
            return _vis.guide_beam_spot_area(_hw._camera, _hw._laser, n_avg=3)

        n_unmeasured = 0

        best_z = cur_z
        best_area = float('inf')
        direction = 1
        step_count = 0
        global_best_area = float('inf')
        global_best_z = cur_z
        z_scores: list = []
        sweep_state = 'init'

        while sweep_state != 'done':
            pos_now = _hw._stage.get_position()
            z_now = pos_now[2] if pos_now else cur_z

            area = _capture_spot_area()
            if area is None:
                n_unmeasured += 1
                if n_unmeasured >= _AF_UNMEASURED_ABORT:
                    _goto_z(cur_z)
                    _laser_off_quiet()
                    return fail(f"Autofocus could not measure the guide-beam spot at all - the camera returned "
                                f"no frames {n_unmeasured} times in a row. Focus was NOT adjusted and the stage "
                                f"is back where it started. This is not something a retry fixes: check that the "
                                f"camera is streaming (start_camera_stream) and that the guide beam is actually "
                                f"emitting (get_laser_status / set_guide_beam_mode).",
                                z_scores=z_scores,
                                unmeasured_samples=n_unmeasured)
                _goto_z(z_now + direction * step_size)
                continue
            z_scores.append({"z": round(z_now, 4), "area_px": area})

            if 0 < area < global_best_area:
                global_best_area = area
                global_best_z = z_now

            if sweep_state == 'init':
                best_area = area
                best_z = z_now
                _goto_z(best_z + direction * step_size)
                sweep_state = 'check'

            elif sweep_state == 'check':
                step_count += 1
                if area < best_area:
                    best_area = area
                    best_z = z_now
                    _goto_z(best_z + direction * step_size)
                else:
                    direction *= -1
                    step_size /= 2.0
                    if step_size < min_step or step_count >= max_steps:
                        sweep_state = 'done'
                    else:
                        _goto_z(best_z + direction * step_size)

        if global_best_area == float('inf'):
            _goto_z(cur_z)
            _laser_off_quiet()
            return fail("Autofocus finished without ever detecting a guide-beam spot - every sample came back "
                        "with zero spot area, so there was nothing to minimise and no focus was found. The Z "
                        "position is unchanged. Check that the guide beam is actually emitting "
                        "(set_guide_beam_mode, then look with analyze_microscope_image) and that a sample is "
                        "actually under the objective.",
                        z_scores=z_scores,
                        unmeasured_samples=n_unmeasured)

        _goto_z(global_best_z)
        time.sleep(0.2)
        recheck_area = _capture_spot_area()
        check = _focus_check(z_scores, global_best_area, global_best_z, recheck_area)
        laser_aim = locate_laser_aim() if check["verdict"] != "failed" else {"updated": False}
        _laser_off_quiet()

        if check["verdict"] == "failed":
            return fail("Autofocus did NOT find a trustworthy focus: " + "; ".join(check["reasons"]) + ". "
                        f"The stage was left at the best Z seen (z={global_best_z}), but treat the focus as "
                        "unknown: spectra taken here may be weak because of defocus, not because of the sample. "
                        "Check the image (analyze_microscope_image); retry once from a different initial_z if the "
                        "surface looks sharp nearby, otherwise report the focus failure.",
                        optimal_z=global_best_z, best_area_px=global_best_area, recheck_area_px=recheck_area,
                        z_scores=z_scores, focus_check=check,
                        current_position={"x": cur_x, "y": cur_y, "z": global_best_z})

        out = ok(optimal_z=global_best_z,
                 best_area_px=global_best_area,
                 recheck_area_px=recheck_area,
                 step_count=step_count,
                 z_scores=z_scores,
                 current_position={"x": cur_x, "y": cur_y, "z": global_best_z},
                 laser_aim=laser_aim,
                 focus_check=check)
        if check["verdict"] == "uncertain":
            out["warning"] = ("Focus is UNCERTAIN: " + "; ".join(check["reasons"]) + ". Before relying on "
                              "signal strength here, confirm focus with the image or a low-power probe.")
        if n_clamped:
            out["z_limit_hits"] = n_clamped
            out["note"] = (
                f"The search hit the Z travel limit ({STAGE_MIN_Z} to {STAGE_MAX_Z} mm) "
                f"{n_clamped} time(s), so the focus may lie outside the reachable range. "
                "If best_area_px is still large, the sample height is likely off - reposition "
                "the sample or the objective rather than repeating autofocus.")
        return out
    except Exception as e:
        _laser_off_quiet()
        return fail(str(e))


@_serialized("analyze_microscope_image")
def analyze_microscope_image(
    question: Annotated[Optional[str], Field(description="What you want to check in the image (optional). e.g. 'Describe the sample position and brightness'")] = 'Find a specific object in the sample (e.g. a cell) and report its center-point pixel coordinates.',
) -> dict:
    """Capture the microscope camera image for the model.

    The image is resized to the view resolution, so its pixel coordinates can be passed to move_to_pixel.

    Returns:
        dict: ok() with the base64 image and the question, or fail().
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    try:
        import cv2
        frame = _hw._camera.get_latest_frame()
        if frame is None:
            return fail("Failed to acquire frame (check whether streaming is active)")

        frame_bgr = _vis.to_view_bgr(frame)
        height, width = frame_bgr.shape[:2]

        aim_x, aim_y = (int(round(v)) for v in _om.aim_px(width, height))
        shown_bgr = _vis.annotate_aim(frame_bgr)
        ret, buf = cv2.imencode('.png', shown_bgr)
        enhanced_question = (
            f"{question}\n\n"
            f"[The attached image has an original resolution of {width}px wide by {height}px tall. "
            f"When returning pixel coordinates, give exact pixel values based on this resolution.]"
            f"[Note: you return pixel coordinates, which are NOT stage coordinates. To move the "
            f"stage to that location, you must use the move_to_pixel(pixel_x, pixel_y) function.]"
            f"[Overlay drawn by the instrument, NOT part of the sample: the red ring at "
            f"({aim_x},{aim_y}) marks where the laser hits - that exact point is what "
            f"acquire_spectrum measures. Ignore it when describing the specimen.]")

        if not ret:
            return fail("PNG encoding failed")
        img_b64 = _vis.encode_for_model(shown_bgr)
        if img_b64 is None:
            return fail("JPEG encoding failed")

        saved_img = _store_save_preview(buf.tobytes(), tag="microscope")

        extent = None
        try:
            if _hw._stage is not None:
                pos = _hw._stage.get_position()
                extent = _om.scene_extent(float(pos[0]), float(pos[1]))
        except Exception:
            pass
        _store_save_scene(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), extent, {})

        out = ok(image_base64=img_b64,
                 question=enhanced_question,
                 width=width,
                 height=height,
                 extent=extent)
        if saved_img.get("ok"):
            out["image_file"] = saved_img["file_id"]
            out["saved"] = {"title": "Microscope view", "image_url": saved_img["image_url"]}
        else:
            out["image_file_error"] = (
                f"The image was NOT saved to disk ({saved_img.get('error')}), so you cannot "
                f"view it again later. Extract everything you need from it in this turn.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("move_to_pixel")
def move_to_pixel(
    pixel_x: Annotated[int, Field(description=f'Image X pixel coordinate (0 - {CAMERA_WIDTH})')],
    pixel_y: Annotated[int, Field(description=f'Image Y pixel coordinate (0 - {CAMERA_HEIGHT})')],
) -> dict:
    """Move the stage so that the given image pixel comes under the laser spot.

    Returns:
        dict: Result of move_stage, or fail().
    """
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    try:
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")
        tx, ty = _om.pixel_to_stage(pixel_x, pixel_y, float(pos[0]), float(pos[1]))
        return move_stage(x=round(tx, 4), y=round(ty, 4))
    except Exception as e:
        return fail(str(e))
