"""Instrument routes: connection, status, settings and manual control."""
from __future__ import annotations

import asyncio
import threading
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.service.tools.hw_tools.system_tools import (
    InstrumentBusy, StageSessionDead, connect_camera, connect_laser,
    connect_stage, connection_flags, latest_frame,
)
from backend.service.hw_panel import apply_hw_settings, hardware_status
from backend.controller.web.setups.init_release import connect_ccd_in_background
from backend.controller.web.setups.request_log import log_hw_status
from backend.controller.web.setups.errors import http_error, unwrap
from backend.controller.web.setups.state import (
    AppState, CcdInitState, StateDep, run_in_worker,
)
from backend.controller.web.dto.requests import (
    AcquireSpectrumRequest,
    CameraConnectRequest,
    HwSettingsRequest,
    LaserConnectRequest,
    MovePixelRequest,
)

router = APIRouter(prefix="/api", tags=["hardware"])


async def _run_connect(state: AppState, connect: Callable[[], str]) -> dict:
    """Run a connect operation in a worker and map exceptions to HTTP status codes."""
    try:
        return {"ok": True, "message": await run_in_worker(state, connect)}
    except InstrumentBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except StageSessionDead as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise http_error(e)


@router.post("/camera/connect")
async def camera_connect(body: CameraConnectRequest, state: StateDep):
    """Connect the microscope camera."""
    return await _run_connect(state, lambda: connect_camera(state.hw, body.exposure_ms))


@router.post("/stage/connect")
async def stage_connect(state: StateDep):
    """Connect the stage."""
    return await _run_connect(state, lambda: connect_stage(state.hw))


@router.post("/laser/connect")
async def laser_connect(body: LaserConnectRequest, state: StateDep):
    """Connect the laser."""
    return await _run_connect(state, lambda: connect_laser(state.hw, body.port))


@router.get("/hw/status")
async def hw_status(state: StateDep):
    """Return the full hardware status: connections, CCD temperature and all settings."""
    st = await run_in_worker(state, lambda: hardware_status(state.hw))
    st["ccd_init"] = state.ccd_state
    log_hw_status(st)
    return st


@router.post("/hw/settings")
async def hw_settings(body: HwSettingsRequest, state: StateDep):
    """Apply settings changed in the panel through the same tool functions the agent uses."""
    changes = body.model_dump(exclude_none=True)
    return unwrap(await run_in_worker(state, lambda: apply_hw_settings(changes, state.hw)),
                  on_error=400)


@router.post("/ccd/connect")
async def ccd_connect(state: StateDep):
    """Retry the CCD connection after a failed startup connection."""
    if connection_flags(state.hw)["ccd"]:
        return {"ok": True, "message": "CCD connected"}
    if state.ccd_state == CcdInitState.FAILED:
        threading.Thread(
            target=connect_ccd_in_background, args=(state,), daemon=True, name="ccd-retry",
        ).start()
        return {"ok": True, "message": "CCD re-initialization started"}
    return {"ok": True, "message": "CCD connection in progress (check /api/hw/status)"}


@router.get("/camera/stream")
async def camera_stream(request: Request, state: StateDep):
    """Stream the camera as MJPEG."""
    import cv2
    from backend.common.vision import to_view_bgr

    if not connection_flags(state.hw)["camera"]:
        raise HTTPException(status_code=503, detail="camera not connected")

    async def generate():
        """Yield MJPEG frames until the client disconnects."""
        while True:
            if await request.is_disconnected():
                break
            try:
                frame = await run_in_worker(state, lambda: latest_frame(state.hw))
            except Exception as e:
                print(f"[CAM STREAM] frame error: {e}")
                await asyncio.sleep(0.2)
                continue
            if frame is not None:
                frame = to_view_bgr(frame)
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/stage/move-pixel")
async def stage_move_pixel(body: MovePixelRequest, state: StateDep):
    """Move the stage relative to a pixel clicked in the camera view."""
    from backend.service.tools.hw_tools.stage_tools import move_stage_relative
    from backend.common import optics_map

    dx_mm, dy_mm = optics_map.pixel_to_stage(
        body.px, body.py, 0.0, 0.0,
        width=body.stream_width, height=body.stream_height,
    )
    return unwrap(await run_in_worker(state, lambda: move_stage_relative(dx=dx_mm, dy=dy_mm)))


@router.post("/spectrum/acquire")
async def spectrum_acquire(body: AcquireSpectrumRequest, state: StateDep):
    """Acquire a spectrum directly for the manual measure button (no LLM)."""
    from backend.service.tools.hw_tools.acquire_tools import acquire_spectrum

    args = body.model_dump(exclude_unset=True)
    result = await run_in_worker(state, lambda: acquire_spectrum(**args))
    return unwrap(result)


@router.get("/health")
async def health(state: StateDep):
    """Report server status and instrument connections."""
    return {
        "status": "ok",
        "hardware": connection_flags(state.hw),
        "ccd_init_state": state.ccd_state,
    }
