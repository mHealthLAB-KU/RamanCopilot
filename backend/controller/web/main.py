"""FastAPI server entry point."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.repo.store.spectrum_store import RESULTS_ROOT  # noqa: E402
from backend.controller.web.setups.init_release import lifespan  # noqa: E402
from backend.controller.web.setups.request_log import RequestLogMiddleware  # noqa: E402
from backend.controller.web.controllers import (  # noqa: E402
    agent, files, hardware, kb,
)

app = FastAPI(title="RamanCopilot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(RequestLogMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/api/results", StaticFiles(directory=str(RESULTS_ROOT)), name="results")

app.include_router(hardware.router)
app.include_router(agent.router)
app.include_router(files.router)
app.include_router(kb.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.controller.web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
