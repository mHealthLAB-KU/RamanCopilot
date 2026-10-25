"""Server lifespan: instrument startup and safe shutdown."""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.common.config import llm as cfg
from backend.repo.llm.llm_call.parse import describe
from backend.repo.llm.providers import parse_model
from backend.repo.hw_access.hw_core import (
    init_hardware as rt_init_hardware,
    sync_tool_handles as rt_sync_handles,
)
from backend.controller.web.setups.state import AppState, CcdInitState


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the instruments on startup and release them on shutdown."""
    state = AppState()
    app.state.s = state
    loop = asyncio.get_event_loop()

    print(f"[SERVER] LLM: {describe()}")

    fast_inits = [
        ("STAGE", state.hw._init_stage),
        ("CAM",   state.hw._init_camera),
        ("LASER", state.hw._init_laser),
    ]
    if parse_model(cfg.OLLAMA_MODEL)[0] == "ollama":
        fast_inits.append(("LLM", state.hw._init_ollama))
    for name, fn in fast_inits:
        try:
            await loop.run_in_executor(state.executor, fn)
            print(f"[SERVER] {name} initialized")
        except Exception as e:
            print(f"[WARN]   {name} initialization failed (skipped): {e}")


    rt_init_hardware(
        stage=state.hw.stage,
        laser=state.hw.laser,
        ccd=None,
        camera=state.hw.camera,
    )

    threading.Thread(
        target=connect_ccd_in_background, args=(state,), daemon=True, name="ccd-init",
    ).start()

    print("[SERVER] ready — http://localhost:8000  (CCD connecting)")
    yield

    state.executor.shutdown(wait=False)
    threading.Thread(
        target=state.hw.shutdown, daemon=False, name="hw-shutdown"
    ).start()
    print("[SERVER] shutdown signal received — shutting down after the CCD warms up.")


def connect_ccd_in_background(state: AppState) -> None:
    """Connect the CCD and start cooling; call from a separate thread."""
    try:
        state.ccd_state = CcdInitState.CONNECTING
        state.hw._init_ccd()
        rt_sync_handles(state.hw)
        state.ccd_state = CcdInitState.READY
        print("[SERVER] CCD connected — tool dispatch updated (cooling continues)")
    except Exception as e:
        state.ccd_state = CcdInitState.FAILED
        state.ccd_error = str(e)
        print(f"[ERROR]  CCD initialization failed: {e}")
