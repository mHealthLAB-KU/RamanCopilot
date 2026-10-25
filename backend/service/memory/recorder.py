"""Automatic recording of every measurement that fires the laser."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
import uuid
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
from pydantic import Field

from backend.common.config import memory as mem_cfg
from backend.common.result import fail, ok
from backend.repo.store import DATA_ROOT, memory_db, run_store

CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("mem_call", default=None)
_PENDING: contextvars.ContextVar[dict | None] = contextvars.ContextVar("mem_pending", default=None)

CROP_FRACTION = 20  # crop width and height are 1/CROP_FRACTION of the view, centered on the laser spot
LPIPS_NET, LPIPS_SIZE = "alex", 64
SPOT_RADIUS_MM = 0.002  # shots within this radius count as the same spot


@contextlib.contextmanager
def call_scope(ctx: dict, tool_call_id: str):
    """Bind the current tool call id for the duration of the call.

    Args:
        ctx (dict): Turn context.
        tool_call_id (str): Identifier of the tool call.
    """
    tok = CALL.set({"session_id": run_store.current().get("session_id") or None,
                    "run_id": ctx.get("mem_run_id"), "skill_version": ctx.get("mem_skill_version"),
                    "tool_call_id": tool_call_id or f"agent:{uuid.uuid4()}"})
    try:
        yield
    finally:
        CALL.reset(tok)


def set_sample_context(
    sample_name: Annotated[str, Field(description="The material name only, exactly as you would write it next time, e.g. 'graphene', 'riboflavin', 'polystyrene'. Past records and learned rules are found by this name (case-insensitive exact match), so do NOT put the substrate, preparation or a description in it ('graphene on Si wafer, drop-cast' would never match 'graphene' again).")],
    source: Annotated[Literal["user", "agent"], Field(description="Who said WHAT THE SAMPLE IS: 'user' if the user told you; 'agent' if it is your own guess. This is about the sample name only - where the band positions came from goes in bands_source.")],
    expected_bands: Annotated[Optional[list[float]], Field(description="Raman shifts (cm-1) where this sample's characteristic bands should appear, taken from the source in bands_source (not from memory). Omit if you have no sourced positions yet.")] = None,
    band_half_width: Annotated[Optional[list[float]], Field(description="Position tolerance (cm-1) for each expected band: how far from the listed centre the band top may sit (literature scatter + calibration). One value per band, or a single value for all. Default 10. The axis is already corrected to Si 520.7, so do not widen it for axis drift. Make it wide enough to cover the whole literature range for the band: if the true top lies outside centre ± half-width, the band is reported as not detected.")] = None,
    band_fwhm: Annotated[Optional[list[float]], Field(description="Expected full width at half maximum (cm-1) of each band, from the same source. One value per band, or a single value for all. Broad bands (e.g. graphene-oxide D/G ~60-100, iron-oxide phonons ~40-80) MUST be given a realistic width or they are judged against their own shoulders and missed. Default 2 x half-width.")] = None,
    bands_source: Annotated[Optional[str], Field(description="Where the band positions/widths came from. Only two values are accepted: 'user' (the user wrote these numbers in this session) or 'web:<url> [<url> ...]' (pages that web_search returned in this session). Required with expected_bands. 'kb:...' or your own knowledge is rejected.")] = None,
    marker_bands: Annotated[Optional[list[float]], Field(description="Optional: band positions (cm-1) of a product you must watch for - e.g. a laser-induced phase (maghemite -> hematite), burnt carbon (D/G), or a contaminant. They are judged on every shot exactly like expected_bands but reported with role='marker', so their appearance after irradiation is visible.")] = None,
    marker_label: Annotated[Optional[str], Field(description="Name of what the marker bands indicate, e.g. 'hematite' or 'amorphous carbon'.")] = None,
    marker_half_width: Annotated[Optional[list[float]], Field(description="Position tolerance (cm-1) per marker band, or one value for all. Default 10.")] = None,
    marker_fwhm: Annotated[Optional[list[float]], Field(description="Expected FWHM (cm-1) per marker band, or one value for all.")] = None,
    marker_source: Annotated[Optional[str], Field(description="Source of the marker bands, same rule as bands_source ('user' or 'web:<url>').")] = None,
) -> dict:
    """Declare the sample context copied into later measurement records.

    Returns:
        dict: ok() with the declared sample and bands, or fail().
    """
    from backend.service.memory import provenance
    name = (sample_name or "").strip()
    if not name:
        return fail("sample_name is required.")

    def per_band(vals, n, default, what):
        if not vals:
            return [default] * n if default is not None else [None] * n
        vals = [float(v) for v in vals]
        if any(v <= 0 for v in vals):
            raise ValueError(f"{what} must be > 0")
        if len(vals) == 1:
            return vals * n
        if len(vals) != n:
            raise ValueError(f"{what} needs one value per band ({n}) or a single value; got {len(vals)}")
        return vals

    def build(centers, hw, fw, src, role, label=None):
        if not centers:
            return None, None
        if not src:
            raise ValueError(f"a source is required for {role} bands ('user' or 'web:<url>').")
        okk, why = provenance.check(src, [float(c) for c in centers])
        if not okk:
            raise ValueError(why)
        hws = per_band(hw, len(centers), 10.0, f"{role} half-width")
        fws = per_band(fw, len(centers), None, f"{role} fwhm")
        out = []
        for c, h, f in zip(centers, hws, fws):
            b = {"center_cm1": float(c), "half_width_cm1": h, "fwhm_cm1": f if f else 2 * h}
            if role == "marker":
                b["role"] = "marker"
                if label:
                    b["label"] = label
            out.append(b)
        return out, why

    try:
        bands, bsrc = build(expected_bands, band_half_width, band_fwhm, bands_source, "expected")
        markers, msrc = build(marker_bands, marker_half_width, marker_fwhm, marker_source, "marker",
                              (marker_label or "").strip() or None)
    except ValueError as e:
        return fail(str(e))
    _SAMPLE_CTX[_session_key()] = {"sample_name": name, "sample_source": source,
                                   "expected_bands": bands, "expected_bands_source": bsrc,
                                   "marker_bands": markers, "marker_bands_source": msrc}
    return ok(sample_name=name, source=source, expected_bands=bands, bands_source=bsrc,
              marker_bands=markers, marker_source=msrc,
              note="Recorded. Every measurement from now on carries this context until you declare another. "
                   "Each shot reports band_check: bands are judged after the Si reference spectrum is subtracted "
                   "and the axis is shifted onto Si 520.7; a band whose window touches the Si line is reported "
                   "as detected=null (cannot be judged).")


_SAMPLE_CTX: dict[str, dict] = {}


def _session_key() -> str:
    return run_store.current().get("label") or "_unassigned"


def sample_ctx() -> dict:
    """Return a copy of the sample context of the current session.

    Returns:
        dict: The declared context, or {'sample_source': 'unknown'}.
    """
    return dict(_SAMPLE_CTX.get(_session_key()) or {"sample_source": "unknown"})


def recorded(fn):
    """Decorate acquire_spectrum so that every call that fires the laser writes one record.

    Args:
        fn (Callable): The acquisition function.

    Returns:
        Callable: The wrapped function.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        tok = _PENDING.set(None)
        result, err = None, None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            err = e
            raise
        finally:
            p = _PENDING.get()
            _PENDING.reset(tok)
            if p is not None and p.get("fired"):
                try:
                    _write(p, result, err)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] memory record failed: {type(e).__name__}: {e}", file=sys.stderr)
    return wrapper


def before_fire(purpose: Optional[str] = None, spot_label: Optional[str] = None) -> None:
    """Record the state just before the instrument is changed.

    Args:
        purpose (str, optional): Reason for this shot and its conditions.
        spot_label (str, optional): Name of the measured spot.
    """
    if not mem_cfg.RECORD:
        return
    from backend.common import optics_map
    from backend.repo.hw_access import hw_core as _hw
    call = CALL.get() or {}
    p = {"fired": False, "record_uid": str(uuid.uuid4()),
         "tool_call_id": call.get("tool_call_id") or f"manual:{uuid.uuid4()}",
         "objective": optics_map.objective(), **sample_ctx(),
         "purpose": (purpose or "").strip()[:500] or None, "spot_label": (spot_label or "").strip()[:60] or None}
    try:
        pos = _hw._stage.get_position() if _hw._stage is not None else None
        if pos is not None:
            p["x_mm"], p["y_mm"] = float(pos[0]), float(pos[1])
            p["z_mm"] = float(pos[2]) if len(pos) > 2 and pos[2] is not None else None
    except Exception:
        pass
    laser = _hw._laser
    p["visible"] = not getattr(laser, "bs_signal", False) and not getattr(laser, "is_on", False)
    p["aim"] = optics_map.aim_px()
    p["before"] = _crop(_view(_hw._camera), p["aim"]) if p["visible"] and mem_cfg.SIDE_EFFECT else None
    _PENDING.set(p)


def artifact_meta() -> dict:
    """Return the ids stored in the saved spectrum files.

    Returns:
        dict: record_uid and tool_call_id, or an empty dict outside a recorded call.
    """
    p = _PENDING.get()
    return {"record_uid": p["record_uid"], "tool_call_id": p["tool_call_id"]} if p else {}


def fired(power_mw: Optional[float]) -> None:
    """Mark that the laser fired.

    Args:
        power_mw (float, optional): Power applied when the laser was turned on.
    """
    p = _PENDING.get()
    if p is not None:
        p["fired"] = True
        p["power_mw"] = power_mw


def after_restore() -> None:
    """Record the state after the optics return to the observation position."""
    p = _PENDING.get()
    if p is None or not p.get("visible") or not mem_cfg.SIDE_EFFECT:
        return
    from backend.common import vision
    from backend.repo.hw_access import hw_core as _hw
    try:
        vision._flush_frames(_hw._camera, 3)
    except Exception:
        pass
    p["after"] = _grab(_hw._camera, p["aim"])


def _write(p: dict, result, err) -> None:
    call = CALL.get() or {}
    uid = p["record_uid"]
    row = {
        "record_uid": uid,
        "session_id": call.get("session_id") or run_store.current().get("session_id") or None,
        "run_id": call.get("run_id"), "skill_version": call.get("skill_version"),
        "tool_call_id": p["tool_call_id"],
        **{k: p.get(k) for k in ("objective", "x_mm", "y_mm", "z_mm", "sample_name", "sample_source",
                                 "expected_bands", "expected_bands_source", "marker_bands",
                                 "purpose", "spot_label")},
    }
    res = result if isinstance(result, dict) and result.get("ok") else None
    if res is not None:
        row.update(_applied(res))
        if p.get("power_mw") is not None:
            row["power_mw"] = p["power_mw"]
        files = (res.get("saved") or {}).get("files") or {}
        row["spectrum_file"] = _rel(files.get("csv"))
        axis, y = _spectrum(res)
        if axis is not None:
            from backend.common import raman_judge
            row["spectrum_qc"] = raman_judge.spectrum_qc(axis, y)
            bands = list(p.get("expected_bands") or []) + list(p.get("marker_bands") or [])
            if bands:
                row["band_values"] = band_values(axis, y, bands)
    else:
        row.update(_device_params(), power_mw=p.get("power_mw"))
        why = (result or {}).get("error") if isinstance(result, dict) else None
        row["error_note"] = (why or (f"{type(err).__name__}: {err}" if err else "no result"))[:300]

    if mem_cfg.SIDE_EFFECT:
        before, after = p.get("before"), p.get("after")
        row["image_before_file"] = _save_crop(before, uid, "before")
        row["image_after_file"] = _save_crop(after, uid, "after")
        if before is not None and after is not None:
            row["lpips"] = lpips_distance(before, after)
    memory_db.insert_measurement(row)


def _applied(res: dict) -> dict:
    """Return the values actually applied to the instrument from a result dict."""
    mode = res.get("mode")
    acc = int(res.get("num_accumulations") or 1)
    kin = int(res.get("kinetic_count") or res.get("num_frames") or 1)
    return {"power_mw": res.get("laser_power_mw"), "exposure_s": res.get("exposure_time"),
            "acq_mode": mode, "num_accumulations": acc if mode in ("accumulate", "kinetic") else 1,
            "kinetic_count": kin if mode == "kinetic" else 1,
            "n_frames": kin * acc if mode == "kinetic" else (acc if mode == "accumulate" else 1),
            "shutter": res.get("shutter"), "trigger_mode": res.get("trigger_mode")}


def _device_params() -> dict:
    """Return the CCD exposure and mode when no result is available."""
    from backend.repo.hw_access import hw_core as _hw
    out = {}
    try:
        out["exposure_s"] = float(_hw._ccd.get_exposure_time())
    except Exception:
        pass
    out["acq_mode"] = getattr(_hw._ccd, "aq_mode", None)
    return out


def _spectrum(res: dict):
    """Return (Raman shift axis, intensity) of a result."""
    if res.get("mode") == "kinetic":
        frames = res.get("frames") or []
        if not frames or "raman_shift_cm-1" not in frames[0]:
            return None, None
        return (np.asarray(frames[0]["raman_shift_cm-1"], float),
                np.mean([f["intensity"] for f in frames], axis=0))
    if "raman_shift_cm-1" not in res or res.get("data") is None:
        return None, None
    return np.asarray(res["raman_shift_cm-1"], float), np.asarray(res["data"], float)


def _rel(path) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def band_values(axis, y, bands: list[dict]) -> list[dict]:
    """Judge each expected band in a spectrum.

    Args:
        axis (array-like): Raman shift axis.
        y (array-like): Intensities.
        bands (list[dict]): Expected bands.

    Returns:
        list[dict]: Result of raman_judge.judge_bands.
    """
    from backend.common import raman_judge
    return raman_judge.judge_bands(axis, y, bands)


def _grab(camera, center=None, n: int = 3):
    """Return a crop of the mean of n frames, or None."""
    return _crop(_view(camera, n), center)


def _view(camera, n: int = 3):
    """Return the mean of n frames at view resolution, or None."""
    if camera is None:
        return None
    from backend.common import vision
    frames = []
    for _ in range(n):
        try:
            f = camera.get_latest_frame()
        except Exception:
            f = None
        if f is not None:
            frames.append(vision.to_view_bgr(f).astype(np.float32))
    return np.mean(frames, axis=0) if frames else None


def _crop(img, center=None):
    """Return a 1/20 crop of a view image around a center (default: the laser spot)."""
    if img is None:
        return None
    from backend.common import optics_map
    h, w = img.shape[:2]
    ch, cw = max(1, h // CROP_FRACTION), max(1, w // CROP_FRACTION)
    ax, ay = center if center is not None else optics_map.aim_px(w, h)
    x0 = min(max(int(round(ax - cw / 2)), 0), w - cw)
    y0 = min(max(int(round(ay - ch / 2)), 0), h - ch)
    return np.clip(img[y0:y0 + ch, x0:x0 + cw], 0, 255).astype(np.uint8)


def _save_crop(img, uid: str, tag: str) -> Optional[str]:
    if img is None:
        return None
    import cv2
    d = run_store.session_dir() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{uid}_{tag}.png"
    cv2.imwrite(str(path), img)
    return _rel(path)


_LPIPS_MODEL = None
_LPIPS_WARNED = False


def lpips_distance(a, b) -> Optional[float]:
    """Compute the LPIPS distance between two image crops.

    Args:
        a (np.ndarray): BGR crop before the shot.
        b (np.ndarray): BGR crop after the shot.

    Returns:
        float or None: Distance, where 0 means identical. None if lpips is not installed.
    """
    global _LPIPS_MODEL, _LPIPS_WARNED
    try:
        import cv2
        import lpips
        import torch
    except ImportError:
        if not _LPIPS_WARNED:
            print("[warn] lpips not installed - side-effect similarity recorded as NULL (pip install lpips)", file=sys.stderr)
            _LPIPS_WARNED = True
        return None
    if _LPIPS_MODEL is None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _LPIPS_MODEL = lpips.LPIPS(net=LPIPS_NET, verbose=False).eval()

    def t(img):
        rgb = cv2.cvtColor(cv2.resize(img, (LPIPS_SIZE, LPIPS_SIZE), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]

    with torch.no_grad():
        return round(float(_LPIPS_MODEL(t(a), t(b)).item()), 6)
