"""Application state and its injection into routes."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Callable, Optional, TypeVar

from fastapi import Depends, Request

from backend.repo.hw_access.hao.hardware_manager import HardwareManager, get_manager


class CcdInitState(str, Enum):
    """Progress stage of the CCD connection."""
    IDLE       = "idle"
    CONNECTING = "connecting"
    READY      = "ready"
    FAILED     = "failed"


@dataclass
class AppState:
    """State kept for the lifetime of the server."""
    hw: HardwareManager = field(default_factory=get_manager)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=4)
    )
    ccd_state: CcdInitState = CcdInitState.IDLE
    ccd_error: Optional[str] = None


def get_state(request: Request) -> AppState:
    """Return the application state for a request."""
    return request.app.state.s


StateDep = Annotated[AppState, Depends(get_state)]


T = TypeVar("T")


async def run_in_worker(state: AppState, fn: Callable[[], T]) -> T:
    """Run a blocking function in a worker thread and await its result."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(state.executor, fn)
