"""Agent tools for attached files and session outputs. Does not import hardware modules."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field

from backend.repo.store.upload_store import ALLOWED_SUFFIXES, inspect_upload, list_uploads, save_upload
from backend.common.result import fail, ok
from backend.service.tools.schema import call_with, tool_schema


def list_uploaded_files(
    date: Annotated[Optional[str], Field(description="Upload date 'YYYY-MM-DD'. If omitted, today.")] = None,
) -> dict:
    """List attached files.

    Returns:
        dict: ok() with file_id, name, kind and size of each file, or fail().
    """
    date = (date or "").strip() or None
    try:
        items = list_uploads(date)
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    if not items:
        return ok(files=[],
                  note="No files have been attached to this chat. Ask the user to attach one, "
                       "or proceed with instrument measurement instead.")
    return ok(count=len(items), files=items)


_MAX_OPEN_BYTES = 4 * 1024 * 1024  # size limit for images and text files


def open_file(
    file_id: Annotated[str, Field(description="A file_id exactly as a listing or an earlier tool result gave it, e.g. 'uploads:2026-07-24/113045_sample.csv', 'results:2026-08-12/_microscope_153726.png', 'runs:<session>/spectra/01_corrected.csv'.")],
    question: Annotated[Optional[str], Field(description="Images only: what you want to check in the picture (optional).")] = None,
    sheet: Annotated[Optional[str], Field(description="Excel only: sheet name to read. If omitted, the first sheet.")] = None,
    max_rows: Annotated[Optional[int], Field(ge=1, le=20, description="Tables only: how many leading rows to preview (1-20). Default 5.")] = None,
) -> dict:
    """Open a stored file and return content according to its kind.

    Images are returned as base64, tables as a summary, and JSON and text files as their content.

    Returns:
        dict: ok() with kind and content, or fail().
    """
    import json as _json

    from backend.repo.store.paths import kind_of, try_resolve

    path, why = try_resolve(file_id)
    if path is None:
        return fail(why)

    file_id = str(file_id or "").strip()
    kind = kind_of(path)
    size = path.stat().st_size
    if kind != "table" and size > _MAX_OPEN_BYTES:
        return fail(f"'{file_id}' is too large to open ({size / 1048576:.1f} MB, "
                    f"limit {_MAX_OPEN_BYTES // 1048576} MB).")

    if kind == "image":
        import cv2
        import numpy as np

        from backend.common.vision import encode_for_model
        try:
            img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return fail(f"Failed to read the image: {type(e).__name__}: {e}")
        img_b64 = encode_for_model(img) if img is not None else None
        if img_b64 is None:
            return fail(f"Failed to decode the image '{file_id}'.")
        question = str(question or "").strip() or "Stored image."
        return ok(kind="image",
                  file_id=file_id,
                  bytes=size,
                  image_base64=img_b64,
                  question=f"{question}\n\n[This is a stored image ({file_id}), not a live view. "
                           f"The stage may have moved since it was captured.]")

    if kind == "json":
        try:
            doc = _json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return fail(f"Failed to read the JSON record: {type(e).__name__}: {e}")
        return ok(kind="json", file_id=file_id, bytes=size,
                  record=doc if isinstance(doc, dict) else {"value": doc})

    if kind == "text":
        try:
            body = path.read_text(encoding="utf-8")
        except Exception as e:
            return fail(f"Failed to read the text file: {type(e).__name__}: {e}")
        if file_id.startswith("memory:"):
            return ok(kind="text", file_id=file_id, bytes=size, text=body,
                      note="This file holds the rules for every sample together. Each rule names the sample "
                           "it applies to; apply only the rules for your sample.")
        return ok(kind="text", file_id=file_id, bytes=size, text=body)

    if kind == "table":
        sheet = (sheet or "").strip() or None
        try:
            max_rows = int(max_rows or 5)
        except (TypeError, ValueError):
            max_rows = 5
        try:
            out = inspect_upload(file_id, sheet=sheet, max_rows=max_rows)
        except FileNotFoundError as e:
            return fail(str(e), hint="Check the exact file_id with list_uploaded_files.")
        except ValueError as e:
            return fail(str(e))
        except Exception as e:
            return fail(f"Failed to read the file: {type(e).__name__}: {e}")
        out["kind"] = "table"
        return out

    return fail(f"'{file_id}' has an extension this tool cannot open ({path.suffix or 'none'}). "
                f"Openable types: {', '.join(sorted(ALLOWED_SUFFIXES | {'.json', '.md'}))}.")


def list_session_artifacts(
    kind: Annotated[Optional[Literal["spectra", "figure", "measurement"]], Field(description="Filter by kind: 'spectra' for saved spectra, 'figure' for plots. Omit to list everything.")] = None,
) -> dict:
    """List the outputs created in the current session.

    Returns:
        dict: ok() with the session and its artifacts, or fail().
    """
    from backend.repo.store import run_store

    kind = (kind or "").strip() or None
    cur = run_store.current()
    try:
        arts = run_store.list_artifacts(kind)
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    if not arts:
        return ok(session=cur["label"],
                  artifacts=[],
                  note="You have not saved anything in this session yet. "
                       "Save computed spectra with save_result inside run_analysis.")
    return ok(session=cur["label"], session_dir=cur["rel_dir"], count=len(arts), artifacts=arts)


def _run_analysis(args: dict) -> dict:
    """Run the analysis sandbox (works without connected hardware)."""
    from backend.service.analyse.analysis_sandbox import run_analysis

    try:
        return run_analysis(
            code=args.get("code"),
            date=args.get("date"),
            names=args.get("names"),
            title=args.get("title"),
            file_ids=args.get("file_ids"),
        )
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


FILE_TOOLS = [
    tool_schema(
        list_uploaded_files,
        "List the files the user attached to the chat (csv, excel, txt, and images). Returns each file's file_id, filename, `kind` and size - not its contents. Open any of them with open_file - it works out the file kind for you. Call this first when the user mentions an attached/uploaded file, or when you are asked to analyze data you did not measure yourself. It touches no hardware and is free of side effects.",
    ),
    tool_schema(
        open_file,
        "Open any stored file by its file_id and get back whatever that file can tell you. "
        "One tool for every kind - you do not have to work out which reader applies:\n"
        "- an IMAGE (.png/.jpg) is shown to you, so you can read pixel coordinates off it "
        "exactly as when it was first captured;\n"
        "- a TABLE (.csv/.xlsx/.txt/...) comes back as a STRUCTURE summary: row/column counts, "
        "column names, numeric-or-text per column, min/max/mean, the first few rows, and (for "
        "Excel) the sheet names. It does NOT return the full data and interprets nothing for "
        "you - you decide whether some numeric column is a Raman shift in cm-1, an intensity, a "
        "stage coordinate, or unrelated metadata, then use run_analysis with file_ids to compute "
        "on the full data;\n"
        "- a JSON record (a saved measurement or measurement point) comes back as its fields.\n"
        "The reply always carries `kind` so you know which of these you got. "
        "Use it for files the user attached AND for anything produced earlier in this session: "
        "a microscope view you captured in an earlier turn (the picture itself is dropped from "
        "the conversation when a turn ends, but the file stays), a grid preview, a saved "
        "spectrum, a measurement record. "
        "Opening an image does NOT take a new picture - it shows the sample as it was at capture "
        "time, so if the stage has moved since, call analyze_microscope_image instead. "
        "It touches no hardware and has no side effects, so call it freely.",
    ),
    tool_schema(
        list_session_artifacts,
        "List the files YOU have produced in this session (processed spectra you saved with "
        "save_result inside run_analysis, measurement-point records, and figures), in the order "
        "you saved them. "
        "Each entry has a `path` you can read back with open_file('<path>'). "
        "Use it when a task builds on something you saved earlier in this conversation, when you "
        "need to confirm a save actually happened, or when you must report where your output went. "
        "It touches no hardware and has no side effects.",
    ),
]

FILE_DISPATCH = {
    "list_uploaded_files":     lambda a: call_with(list_uploaded_files, a),
    "open_file":               lambda a: call_with(open_file, a),
    "run_analysis":            _run_analysis,
    "list_session_artifacts":  lambda a: call_with(list_session_artifacts, a),
}
