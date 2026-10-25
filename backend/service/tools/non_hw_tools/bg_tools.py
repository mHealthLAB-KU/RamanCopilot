"""Loading of saved spectrum CSV files."""

from __future__ import annotations

import csv

from backend.common.result import fail, ok
from pydantic import Field
from typing import Annotated


def _resolve_data_path(source: str):
    """Resolve a file reference to a path; return (path, error)."""
    from backend.repo.store.paths import try_resolve

    return try_resolve(source)


def load_spectrum(
    filename: Annotated[str, Field(description="A file_id, in the form '<area>:<path>' where area is uploads, results or runs - exactly as the listing tools return it (e.g. 'uploads:2026-08-07/N05.csv', 'runs:<session>/spectra/01_corrected.csv'). Paths without the area prefix and absolute paths are also accepted.")],
) -> dict:
    """Load a saved spectrum CSV or an uploaded file.

    Returns:
        dict: ok() with the axis and intensity columns, or fail().
    """
    try:
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath, why = _resolve_data_path(filename)
        if filepath is None:
            return fail(why)

        comments: dict = {}
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.read().splitlines()
        body = []
        for ln in lines:
            if not body and ln.lstrip().startswith("#"):
                bits = ln.lstrip().lstrip("#").strip().split(",", 1)
                if len(bits) == 2:
                    comments[bits[0].strip()] = bits[1].strip()
                continue
            body.append(ln)

        reader = csv.DictReader(body)
        headers = reader.fieldnames or []
        rows = list(reader)
        if "frame_index" in headers:
            n_frames = len({r.get("frame_index") for r in rows})
            return fail(f"{filepath.name} is a Kinetic measurement ({n_frames} frames stored as "
                        f"frame_index,pixel_index,intensity). load_spectrum only reads 1D spectra "
                        f"(Single/Accumulate) and would concatenate the frames into one wrong array. "
                        f"Analyse it with run_analysis instead: kinetic measurements arrive there as "
                        f"spectra[i]['frames'], a (n_frames, n_pixels) array.")
        col = next((c for c in ("intensity", "corrected_intensity") if c in headers), None)
        if col is None:
            return fail(f"No 'intensity' (or 'corrected_intensity') column in "
                        f"{filepath.name}. Columns: {headers}")

        intensity = [float(r[col]) for r in rows]
        result: dict = ok(filename=str(filepath),
                          num_points=len(intensity),
                          headers=headers,
                          intensity_column=col,
                          intensity=intensity)
        if comments:
            if "laser_power_mw" not in comments and comments.get("laser_power_pct"):
                from backend.repo.store.spectrum_store import power_mw_of
                try:
                    comments["laser_power_mw"] = str(power_mw_of({"laser_power_pct": float(comments["laser_power_pct"])}))
                except ValueError:
                    pass
            result["metadata"] = comments
        if "raman_shift_cm-1" in headers:
            result["raman_shift_cm-1"] = [float(r["raman_shift_cm-1"]) for r in rows]
        if "wavelength_nm" in headers:
            result["wavelength_nm"] = [float(r["wavelength_nm"]) for r in rows]
        if "background_intensity" in headers:
            result["background_intensity"] = [float(r["background_intensity"]) for r in rows]
        return result
    except Exception as e:
        return fail(str(e))
