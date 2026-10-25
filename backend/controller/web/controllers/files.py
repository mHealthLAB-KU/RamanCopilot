"""Routes for uploading and listing attached data files."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.controller.web.setups.errors import http_error

router = APIRouter(prefix="/api", tags=["files"])


@router.post("/files/upload")
async def files_upload(file: UploadFile = File(...)):
    """Store an uploaded data file in the uploads folder of today and return its file_id."""
    from backend.service.tools.non_hw_tools.file_tools import ALLOWED_SUFFIXES, save_upload

    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="no file name")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(f"unsupported format. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))} "
                    f"(papers and protocols go to /api/kb/upload)"),
        )
    try:
        data = await file.read()
        return save_upload(name, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise http_error(e, detail=f"save failed: {e}")


@router.get("/files")
async def files_list(date: Optional[str] = None):
    """List files uploaded on a given date (default today)."""
    from backend.service.tools.non_hw_tools.file_tools import list_uploads
    return {"ok": True, "files": list_uploads(date)}
