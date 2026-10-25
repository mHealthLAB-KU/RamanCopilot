"""Agent tools whose implementation is in the service layer."""

from __future__ import annotations

from backend.service.analyse.analysis_sandbox import run_analysis as _run_analysis
from backend.service.search.web_search import web_search as _web_search
from backend.repo.store.spectrum_store import aggregate_spectra_csv as _store_aggregate_csv, bundle_results as _store_bundle_results, combine_spectra as _store_combine_spectra, list_results as _store_list_results
from backend.common.result import fail, ok
from backend.service.tools.schema import INTERNAL
from pydantic import Field
from typing import Annotated, Literal, Optional


def list_results(
    date: Annotated[Optional[str], Field(description="Date to query 'YYYY-MM-DD'. If omitted, today.")] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to consider. 'session' (default) = only the ones measured in THIS session, which is almost always what you want. 'all' = every session saved that day; use it only when the request is explicitly about combining work from earlier, separate sessions.")] = None,
) -> dict:
    """List saved measurements.

    Returns:
        dict: Result of spectrum_store.list_results.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_list_results(**given)


def combine_spectra(
    date: Annotated[Optional[str], Field(description="Target date 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to combine (check with list_results). If omitted, the whole date.')] = None,
    max_cols: Annotated[Optional[int], Field(ge=1, description='Number of grid columns. Default 4.')] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to consider. 'session' (default) = only the ones measured in THIS session, which is almost always what you want. 'all' = every session saved that day; use it only when the request is explicitly about combining work from earlier, separate sessions.")] = None,
    out_name: Annotated[Optional[str], Field(json_schema_extra=INTERNAL, description='Internal: output file stem. Defaults to combined_<HHMMSS>.')] = None,
) -> dict:
    """Render several measurements into one grid image.

    Returns:
        dict: Result of spectrum_store.combine_spectra.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_combine_spectra(**given)


def aggregate_spectra_csv(
    date: Annotated[Optional[str], Field(description="Target date 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to organize. If omitted, all of yours from that date.')] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to consider. 'session' (default) = only the ones measured in THIS session, which is almost always what you want. 'all' = every session saved that day; use it only when the request is explicitly about combining work from earlier, separate sessions.")] = None,
    out_name: Annotated[Optional[str], Field(json_schema_extra=INTERNAL, description='Internal: output file stem. Defaults to summary_<HHMMSS>.')] = None,
) -> dict:
    """Write a summary CSV with one row per measurement.

    Returns:
        dict: Result of spectrum_store.aggregate_spectra_csv.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_aggregate_csv(**given)


def web_search(
    query: Annotated[str, Field(description="Search query. e.g. 'raman baseline correction asymmetric least squares'")],
    max_results: Annotated[Optional[int], Field(ge=1, le=10, description='Number of results to fetch (1-10). Default 5.')] = None,
) -> dict:
    """Search the web. Used only by Ollama models.

    Returns:
        dict: Result of service.search.web_search.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _web_search(**given)


def run_analysis(
    code: Annotated[str, Field(description="Python analysis code to run. Use spectra, np, plt directly. e.g. compute each spectrum's peak intensity and draw a peak map as an (x,y) scatter. If the task asks you to save a computed spectrum, call save_result('name', corrected_intensity, raman_shift=x) at the end of this code rather than printing the array. KEEP EACH CALL SHORT - aim for 40 lines or fewer. This code travels to the sandbox as a single JSON string, and a long block (many escaped quotes and newlines) is the most common way for a call to be lost in transit: the call silently never arrives and the task ends with no answer. Do not write one large end-to-end script. Split the work and call this tool several times - e.g. (1) load the data and print its shape and column names, (2) do one computation step and print a short summary, (3) produce the final numbers. Nothing carries over between calls: every call runs in a fresh process, so each one must rebuild what it needs from spectra/files, or read back an intermediate you wrote earlier with save_result (pass the path it returned as a file_id in file_ids).")],
    date: Annotated[Optional[str], Field(description="Measurement date to analyze 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to analyze (check with list_results). If omitted, the whole date.')] = None,
    file_ids: Annotated[Optional[list[str]], Field(description='file_ids of attached files to load into the `files` variable (get them from list_uploaded_files). If omitted, no attached file is loaded.')] = None,
    title: Annotated[Optional[str], Field(description='Title to attach to the result figure.')] = None,
    timeout_sec: Annotated[Optional[int], Field(json_schema_extra=INTERNAL, description='Internal: sandbox wall-clock limit (s). Deliberately not model-settable - a runaway script must not be able to raise its own timeout.')] = None,
) -> dict:
    """Run analysis code in the sandbox.

    This declaration provides the schema. The call is executed through file_tools.

    Returns:
        dict: Result of analysis_sandbox.run_analysis.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _run_analysis(**given)


def bundle_results(
    date: Annotated[Optional[str], Field(description="Target date 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to bundle. If omitted, all of yours from that date.')] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to bundle. 'session' (default) = only the ones measured in THIS session. 'all' = every session saved that day; use it only when the request is explicitly about earlier, separate sessions.")] = None,
    include: Annotated[Optional[list[str]], Field(json_schema_extra=INTERNAL, description='Internal: which file kinds to bundle. Defaults to png+csv+json.')] = None,
) -> dict:
    """Bundle saved result files into one zip file.

    Returns:
        dict: Result of spectrum_store.bundle_results.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_bundle_results(**given)


def identify_material(
    source: Annotated[Optional[str], Field(description="Spectrum to identify. 'last': the most recent acquire_spectrum() result (default). Otherwise a JSON or CSV file_id in the form '<area>:<path>' (e.g. 'results:2026-09-06/<session>/1408_x37.csv'), exactly as the listing tools return it. A Kinetic measurement is rejected - it has no single intensity array.")] = 'last',
    top_k: Annotated[Optional[int], Field(ge=1, le=20, description='How many candidate minerals to return (1-20). Default 5. One row per mineral - the library holds many samples of the same mineral (21 quartz spectra, for instance) and only the best-scoring one of each is shown, so the list is five different candidates rather than five samples of one.')] = 5,
    algorithm: Annotated[Optional[Literal['correlation', 'combined', 'peak', 'multi_window', 'mineral_vibration', 'dtw']], Field(description="Similarity measure. 'correlation' (default) compares the whole spectrum shape and is the one to start with. 'peak' scores peak positions instead of shape - use it when the spectrum is noisy or the background is uneven. 'combined' mixes correlation, DTW and peaks and is the most thorough. 'multi_window' and 'mineral_vibration' weight diagnostic frequency bands. Every option runs a fast correlation pass over the whole library first and only re-scores the top 50, so the cost difference is small.")] = 'correlation',
    subtract_background: Annotated[Optional[bool], Field(description="Remove the fluorescence background from your spectrum before comparing (ALS). Default true, and you almost always want it - the library spectra are already background-corrected, so an uncorrected query matches the shape of its own fluorescence hump rather than its peaks. Set false only if you already subtracted the background (e.g. the source is a corrected spectrum you saved from run_analysis).")] = True,
) -> dict:
    """Match a measured spectrum against the RRUFF library.

    Returns:
        dict: ok() with the ranked candidates and score margin, or fail().
    """
    from backend.service.tools.session_state import _sstate
    from backend.service.tools.non_hw_tools.bg_tools import _resolve_data_path, load_spectrum

    source = str(source or "last")
    top_k = int(top_k or 5)
    algorithm = str(algorithm or "correlation")
    subtract_background = True if subtract_background is None else bool(subtract_background)

    intensity, raman_shift = None, None

    if source == "last":
        last = _sstate()["last_spectrum"]
        if last is None:
            return fail("No saved spectrum. Call acquire_spectrum() first, or pass a file_id "
                        "from list_results as source.")
        if "data" not in last:
            return fail("The last spectrum is in Kinetic mode. This tool compares a single "
                        "spectrum; average the frames in run_analysis and save that instead.")
        intensity = last["data"]
        raman_shift = last.get("raman_shift_cm-1")
    else:
        filepath, why = _resolve_data_path(source)
        if filepath is None:
            return fail(why)
        loaded = load_spectrum(str(filepath))
        if not loaded.get("ok"):
            return fail(f"File load error: {loaded.get('error')}")
        intensity = loaded["intensity"]
        raman_shift = loaded.get("raman_shift_cm-1")

    if not intensity:
        return fail("The spectrum intensity array is empty.")
    if not raman_shift:
        return fail("This spectrum has no raman_shift_cm-1 axis, so it cannot be compared with "
                    "the library. Acquire it in Raman shift mode, or pass a CSV that has that column.")
    if len(raman_shift) != len(intensity):
        return fail(f"Axis length {len(raman_shift)} does not match intensity length {len(intensity)}.")

    from backend.service.knowledge import identify

    try:
        hits = identify(raman_shift, intensity, top_k=top_k, algorithm=algorithm,
                        subtract_background=subtract_background)
    except Exception as e:
        return fail(f"Identification failed: {e}")

    if not hits:
        return ok(results=[], note="The library returned no candidate. Report the peak positions "
                                   "you measured and identify the sample from them yourself.")

    margin = round(hits[0]["score"] - hits[1]["score"], 4) if len(hits) > 1 else None
    return ok(results=hits,
              algorithm=algorithm,
              background_subtracted=bool(subtract_background),
              score_margin=margin,
              note="Scores are relative similarity, not probability. Judge the identification from "
                   "the gap to the runner-up (score_margin) and from whether the candidate's "
                   "chemistry fits what you know about the sample - a top score close to the others "
                   "means the library has no good match, not that the first row is right.")
