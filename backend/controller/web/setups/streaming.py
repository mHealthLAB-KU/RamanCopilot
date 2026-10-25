"""Bridge from a synchronous event generator to a Server-Sent Events response."""
from __future__ import annotations

import asyncio
import json
from typing import Callable, Optional

from fastapi.responses import StreamingResponse

from backend.controller.web.setups.state import AppState

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _frame(event_type: str, payload: dict) -> str:
    """Format one SSE block."""
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_response(
    state: AppState,
    produce: Callable[[Callable[[dict], None]], None],
    *,
    first_event: Optional[tuple[str, dict]] = None,
) -> StreamingResponse:
    """Run produce in a worker thread and stream the events it emits as SSE."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _worker():
        try:
            produce(emit)
        except Exception as e:
            emit({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    state.executor.submit(_worker)

    async def event_gen():
        if first_event is not None:
            yield _frame(first_event[0], first_event[1])
        while True:
            event = await queue.get()
            if event is DONE:
                break
            yield _frame(event.get("type", "message"), event)

    return StreamingResponse(
        event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS,
    )
