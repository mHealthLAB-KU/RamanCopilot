"""Knowledge-base routes for status and cache reload."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["kb"])


@router.get("/kb/status")
async def kb_status_endpoint():
    """Report how many entries the knowledge base has loaded."""
    from backend.service.knowledge import db_status
    return db_status()


@router.get("/kb/search")
async def kb_search_endpoint():
    """Deprecated endpoint; always returns 410."""
    raise HTTPException(
        status_code=410,
        detail="The KB is now a spectrum library, so a keyword query has no meaning here. "
               "Run `python -m backend.service.knowledge.kb_spectrum_search` to search it, "
               "or call the identify_material tool.")


@router.post("/kb/reload")
async def kb_reload_endpoint():
    """Clear the knowledge-base cache so the next query reloads it from disk."""
    from backend.service.knowledge import db_status, reload_db
    reload_db()
    return {"ok": True, "message": "KB cache cleared", "status": db_status()}
