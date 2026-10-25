"""Agent routes used by the chat interface."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import backend.common.config.agent as agent_cfg
import backend.common.config.llm as llm_cfg
from backend.controller.web.setups.agent_module import select_agent_module
from backend.controller.web.setups.state import StateDep, run_in_worker
from backend.controller.web.setups.streaming import sse_response
from backend.controller.web.dto.requests import ExperimentRequest
from backend.controller.web.setups.errors import http_error

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/experiment/run")
async def experiment_run(body: ExperimentRequest, state: StateDep):
    """Run the agent once synchronously without session history."""
    mod, _ = select_agent_module(body.agent)
    try:
        return await run_in_worker(
            state, lambda: mod.run_experiment(body.message, body.session_id or ""))
    except Exception as e:
        raise http_error(e)


@router.post("/experiment/stream")
async def experiment_stream(body: ExperimentRequest, state: StateDep) -> StreamingResponse:
    """Run the agent and stream tool progress and chat events over SSE."""
    mod, _ = select_agent_module(body.agent)

    def produce(emit):
        try:
            for event in mod.stream_experiment(body.message, body.session_id or ""):
                emit(event)
        except Exception as e:
            emit({"type": "error", "detail": str(e)})

    return sse_response(state, produce)


@router.get("/agents/health")
async def agents_health(agent: str = ""):
    """Report the agent system status for the selected architecture."""
    mod, name = select_agent_module(agent)
    return {
        "status": "ok",
        "agents": [f"single_agent_{name}"],
        "model": llm_cfg.OLLAMA_MODEL,
        "host": llm_cfg.OLLAMA_HOST,
        "num_ctx": llm_cfg.NUM_CTX,
        "chat_isolated": agent_cfg.CHAT_SESSION_ISOLATED,
        "tools": len(mod.ALL_TOOLS),
    }
