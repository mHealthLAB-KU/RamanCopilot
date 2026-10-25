
from __future__ import annotations

from benchmark.test_pipeline.graders import _util as U


def _one(r):
    """Return the result of one measurement saved in this turn."""
    got = r.new_results
    return got[0]["result"] if got else {}


def _saved(r, n: int):
    got = len(r.new_results)
    return got == n, f"{n} measurement file set(s) (got {got})"


def _xy(r, i: int = 0):
    """Return (x, y) of the i-th saved measurement."""
    got = r.new_results
    if len(got) <= i:
        return [], []
    return r.spectrum(got[i]["base"])


def A01(r):
    yield _saved(r, 1)
    yield r.cites(_one(r).get("max_intensity"), tol=1e-3, rel=True), "cited max intensity"


def A02(r):
    """Shift of the maximum intensity within 500-600 cm-1."""
    yield _saved(r, 1)
    x, y = _xy(r)
    xs, ys = U.in_range(x, y, 500, 600)
    yield len(ys) >= 2, "at least 2 points in 500-600 cm-1"
    if ys:
        top = max(ys)
        ties = [a for a, b in zip(xs, ys) if b == top]
        yield any(r.cites(t, tol=1e-6) for t in ties), \
            f"cited the argmax shift ({len(ties)} tie(s), all accepted)"


def A03(r):
    """5th-order background removal saved as CSV."""
    from benchmark.test_pipeline.inject import read_spectrum_csv
    yield r.called("apply_background_subtraction"), "called apply_background_subtraction"
    yield r.args("apply_background_subtraction").get("poly_order") == 5, "poly_order == 5"
    paths = [p for p in U.saved_paths(r, "spectra") if p.suffix.lower() == ".csv"]
    yield bool(paths), "result CSV saved"
    ok = False
    for p in paths:
        head = p.read_text(encoding="utf-8-sig").splitlines()
        cols = next((ln for ln in head if not ln.startswith("#")), "")
        if "background_intensity" in cols and "intensity" in cols and "pixel_index" in cols:
            ok = True
    yield ok, "header == pixel_index, [raman_shift_cm-1,] intensity, background_intensity"


def A04(r):
    """Standard deviation of the maximum intensity over five measurements at one spot."""
    yield _saved(r, 5)
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "stage position unchanged"
    peaks = [float(x["result"].get("max_intensity") or 0) for x in r.new_results]
    yield r.cites_any(U.stdev_both(peaks), tol=1e-3, rel=True), \
        "cited stdev (both ddof 0 and 1 accepted)"


def A05(r):
    """Comparison of the current spot and a spot 0.1 mm away in X."""
    yield _saved(r, 2)
    xs = sorted(float(x["meta"].get("x") or 0) for x in r.new_results)
    yield len(xs) == 2 and abs((xs[1] - xs[0]) - 0.1) <= 0.001, "the two positions differ by 0.1 mm in x"
    peaks = [float(x["result"].get("max_intensity") or 0) for x in r.new_results]
    yield all(r.cites(p, tol=1e-3, rel=True) for p in peaks), "cited both max intensities"


def A06(r):
    """Three measurements plus a one-row-per-measurement summary table."""
    import csv
    from backend.service.store.spectrum_store import RESULTS_ROOT
    yield _saved(r, 3)
    hits = list(RESULTS_ROOT.glob(f"*/_summary_*.csv"))
    hits = [p for p in hits if p in r.files_after and p not in r.files_before]
    yield len(hits) == 1, f"exactly 1 summary CSV (got {len(hits)})"
    if hits:
        rows = list(csv.reader(hits[0].read_text(encoding="utf-8-sig").splitlines()))
        want = ["date", "time", "title", "x", "y", "power_pct", "exposure_s",
                "mode", "max_intensity", "sum_intensity", "peak_raman_shift_cm-1"]
        yield rows and rows[0] == want, "summary header matches"
        yield len(rows) == 4, f"3 data rows (got {max(0, len(rows) - 1)})"


def A07(r):
    """Wavelength (nm) of the point closest to 1000 cm-1."""
    from benchmark.test_pipeline.inject import read_spectrum_csv
    yield _saved(r, 1)
    got = r.new_results
    if not got:
        return
    import csv as _csv
    from pathlib import Path
    lines = Path(got[0]["csv"]).read_text(encoding="utf-8-sig").splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    rows = list(_csv.reader(body))
    head = rows[0]
    yield "wavelength_nm" in head and "raman_shift_cm-1" in head, "both axis columns present"
    if "wavelength_nm" in head and "raman_shift_cm-1" in head:
        iw, ir = head.index("wavelength_nm"), head.index("raman_shift_cm-1")
        pairs = [(abs(float(x[ir]) - 1000.0), float(x[iw])) for x in rows[1:] if len(x) > max(iw, ir)]
        best = min(p[0] for p in pairs)
        yield any(r.cites(w, tol=1e-6) for d, w in pairs if d == best), \
            "cited the wavelength nearest 1000 cm-1 (ties all accepted)"


def A08(r):
    """Trapezoidal integral over 900-1100 cm-1."""
    yield _saved(r, 1)
    x, y = _xy(r)
    xs, ys = U.in_range(x, y, 900, 1100)
    yield len(ys) >= 2, "at least 2 points in the window"
    yield r.cites(U.trapz(xs, ys), tol=1e-6, rel=True), "cited the integral"


def A09(r):
    """Extract and plot the 800-1200 cm-1 range without interpolation."""
    yield _saved(r, 1)
    x, y = _xy(r)
    want_x, want_y = U.in_range(x, y, 800, 1200)
    got = U.agent_spectra(r)
    yield any(U.allclose(gy, want_y) and U.allclose(gx, want_x) for gx, gy in got), \
        "saved window equals the acquired subset exactly (no interpolation)"
    yield bool(U.saved_paths(r, "figure")) or r.called("run_analysis"), "one plot"


def A10(r):
    """Correlation between laser power (1, 3, 5, 10%) and maximum intensity."""
    yield _saved(r, 4)
    got = sorted(r.new_results, key=lambda it: float(it["result"].get("laser_power_pct") or 0))
    powers = [float(x["result"].get("laser_power_pct") or 0) for x in got]
    yield {round(p) for p in powers} == {1, 3, 5, 10}, f"power set (got {sorted(powers)})"
    peaks = [float(x["result"].get("max_intensity") or 0) for x in got]
    yield r.cites(U.pearson(powers, peaks), tol=1e-6), "cited the correlation coefficient"


def A11(r):
    """Maximum intensity at 1001±10 cm-1 after 5th-order background removal."""
    yield _saved(r, 1)
    yield r.args("apply_background_subtraction").get("poly_order") == 5, "poly_order == 5"
    got = U.agent_spectra(r)
    vals = [U.max_in(x, y, 991, 1011) for x, y in got]
    yield any(v is not None and r.cites(v, tol=1e-6, rel=True) for v in vals), \
        "cited the max over 1001+-10 cm-1 of the corrected result"


def A12(r):
    """Mean spectrum of a 3x3 grid scan."""
    yield r.sequence("preview_grid_scan", "run_grid_scan"), "preview_grid_scan before run_grid_scan"
    yield _saved(r, 9)
    xs = [r.spectrum(it["base"])[1] for it in r.new_results]
    yield bool(xs) and len({len(v) for v in xs}) == 1, "the 9 spectra have equal length"
    if xs:
        mean = [sum(col) / len(col) for col in zip(*xs)]
        yield any(U.allclose(y, mean) for _x, y in U.agent_spectra(r)), "saved mean == mean of the 9 points"


def A13(r):
    """Three points 0.02 mm apart in X with a table of maximum intensity at 1001±10 cm-1."""
    yield _saved(r, 3)
    got = sorted(r.new_results, key=lambda it: float(it["meta"].get("x") or 0))
    xs = [float(it["meta"].get("x") or 0) for it in got]
    yield len(xs) == 3 and all(abs((xs[i + 1] - xs[i]) - 0.02) <= 0.001 for i in range(2)), \
        "x spaced evenly by 0.02 mm"
    yield r.unchanged("pos.y", "pos.z", tol=0.001), "y and z unchanged"
    for it in got:
        x, y = r.spectrum(it["base"])
        v = U.max_in(x, y, 991, 1011)
        yield v is not None and r.cites(v, tol=1e-6, rel=True), f"cited the window max of {it['base']}"


def A14(r):
    """Total intensity of a single acquisition versus a 4-accumulation acquisition."""
    yield _saved(r, 2)
    modes = {x["result"].get("mode") for x in r.new_results}
    yield modes == {"single", "accumulate"}, f"mode set (got {sorted(m or '' for m in modes)})"
    acc = next((x["result"] for x in r.new_results
                if x["result"].get("mode") == "accumulate"), {})
    yield acc.get("num_accumulations") == 4, "num_accumulations == 4"
    yield all(r.cites(float(x["result"].get("sum_intensity") or 0), tol=1e-3, rel=True)
              for x in r.new_results), "cited both summed intensities"
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "stage position unchanged"


def A15(r):
    """Mean and standard deviation of per-frame maximum intensity over 10 kinetic frames."""
    yield _saved(r, 1)
    frames = _one(r).get("frames") or []
    yield len(frames) == 10, f"10 frames (got {len(frames)})"
    peaks = [float(f.get("max_intensity") or 0) for f in frames]
    if peaks:
        yield r.cites(sum(peaks) / len(peaks), tol=1e-3, rel=True), "cited the mean"
        yield r.cites_any(U.stdev_both(peaks), tol=1e-3, rel=True), \
            "cited stdev (both ddof 0 and 1 accepted)"


def A16(r):
    """Standard deviation over 1800-2000 cm-1 before and after smoothing."""
    yield _saved(r, 1)
    x, y = _xy(r)
    _nx, raw = U.in_range(x, y, 1800, 2000)
    yield r.cites_any(U.stdev_both(raw), tol=1e-3, rel=True), "cited the source stdev"
    got = U.agent_spectra(r)
    yield bool(got), "smoothed result saved"
    ok_after = ok_lower = False
    for gx, gy in got:
        _n, sm = U.in_range(gx, gy, 1800, 2000)
        cands = U.stdev_both(sm)
        if cands and r.cites_any(cands, tol=1e-3, rel=True):
            ok_after = True
            if min(cands) < max(U.stdev_both(raw) or [0]):
                ok_lower = True
    yield ok_after, "cited the smoothed stdev"
    yield ok_lower, "smoothed stdev < source stdev"


def A17(r):
    """SNR comparison of 3rd- and 7th-order background removal."""
    orders = {a.get("poly_order") for a in r.all_args("apply_background_subtraction")}
    yield orders == {3, 7}, f"poly_order set == {{3,7}} (got {sorted(o or 0 for o in orders)})"
    labels = {a.get("version_label") for a in r.all_args("apply_background_subtraction")}
    yield len(labels) == 2, "two distinct version_labels"
    yield (r.result("list_bg_versions").get("count") or 0) >= 2, "list_bg_versions count >= 2"

    got = U.agent_spectra(r)
    snrs = [s for x, y in got for s in U.snr_both(x, y)]
    yield bool(snrs), "SNR recomputable from both results"
    yield r.cites_any(snrs, tol=0.05, rel=True), "cited SNR (+-5%)"


def A18(r):
    """Exposure (0.5, 1, 2 s) giving the highest SNR."""
    yield _saved(r, 3)
    got = r.new_results
    exps = {round(float(x["result"].get("exposure_time") or 0), 3) for x in got}
    yield exps == {0.5, 1.0, 2.0}, f"exposure set (got {sorted(exps)})"
    best, best_snr = None, -1.0
    for it in got:
        x, y = r.spectrum(it["base"])
        cands = U.snr_both(x, y)
        if cands and max(cands) > best_snr:
            best_snr, best = max(cands), float(it["result"].get("exposure_time") or 0)
    msg = f"reported the SNR argmax condition ({best} s)"
    yield best is not None and r.cites(best, tol=U.DEV_REL, rel=True), msg


def A19(r):
    """3x3 grid with a heatmap of maximum intensity at 1001±10 cm-1."""
    yield r.sequence("preview_grid_scan", "run_grid_scan"), "preview_grid_scan before run_grid_scan"
    yield _saved(r, 9)
    yield bool(U.saved_paths(r, "figure")) or r.called("run_analysis"), "one heatmap"
    vals = []
    for it in r.new_results:
        x, y = r.spectrum(it["base"])
        v = U.max_in(x, y, 991, 1011)
        if v is not None:
            vals.append(v)
    yield len(vals) == 9, f"window max recomputed for all 9 points (got {len(vals)})"


def A20(r):
    """Five measurements at one spot with the mean saved and plotted."""
    yield _saved(r, 5)
    yield r.unchanged("pos.x", "pos.y", tol=0.001), "stage position unchanged"
    ys = [r.spectrum(it["base"])[1] for it in r.new_results]
    yield bool(ys) and len({len(v) for v in ys}) == 1, "the 5 spectra have equal length"
    if ys:
        mean = [sum(col) / len(col) for col in zip(*ys)]
        yield any(U.allclose(y, mean) for _x, y in U.agent_spectra(r)), "saved mean == mean of the 5 spectra"
    yield bool(U.saved_paths(r, "figure")) or r.called("run_analysis"), "plot"
