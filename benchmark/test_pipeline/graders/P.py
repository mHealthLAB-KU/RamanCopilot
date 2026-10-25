
from __future__ import annotations

from benchmark.test_pipeline.graders import _util as U


def P01(r):
    """File with the highest maximum intensity among this session's measurements."""
    yield r.called("list_results"), "called list_results"
    res = r.results
    yield len(res) >= 3, f"3 injected measurements visible (got {len(res)})"
    if res:
        top = max(res, key=lambda it: float(it["result"].get("max_intensity") or 0))
        want = r.item["answer"]["file"].removesuffix(".csv")
        yield want in top["base"], f"true argmax is {want} (setup check)"
        yield r.mentions(want), f"reported {want}"


def P02(r):
    """SNR = maximum / standard deviation over 1800-2000 cm-1."""
    a = r.item["answer"]
    yield r.cites_any([a["snr_ddof1"], a["snr_ddof0"]], tol=0.01, rel=True), \
        "cited SNR (both ddof 0 and 1 accepted)"


def P03(r):
    yield r.cites(r.item["answer"]["ratio_1001_over_1602"], tol=1e-6, rel=True), \
        "cited the 1001/1602 intensity ratio"


def P04(r):
    """Save the mean spectrum of three files."""
    yield U.matches_expected(r, r.item)
    yield len(r.item["inputs"]) == 3, "3 input files (setup check)"


def P05(r):
    """Convert a nm axis to Raman shift (cm-1, 532.021 nm excitation) and save."""
    yield U.matches_expected(r, r.item)


def P06(r):
    """Shift of the largest difference between two files."""
    yield r.cites(r.item["answer"]["shift_of_max_abs_diff"], tol=1e-6), "cited the shift of maximum difference"


def P07(r):
    """Pick the outlier among five spectra."""
    want = r.item["answer"]["outlier_file"].removesuffix(".csv")
    yield r.mentions(want), f"named {want}"
    others = [n.removesuffix(".csv") for n in r.item["inputs"]
              if not n.startswith(want)]
    yield sum(r.mentions(o) for o in others) < len(others), \
        "did not name all the others (it chose one)"


def P08(r):
    """Resample to a uniform 0.5 cm-1 grid."""
    a = r.item["answer"]
    got = U.agent_spectra(r)
    yield bool(got), "a spectrum was saved"
    ok = False
    for x, y in got:
        if len(x) < 2:
            continue
        spacing_ok = all(abs((x[i + 1] - x[i]) - a["spacing"]) < 1e-6 for i in range(len(x) - 1))
        ex_x, ex_y = U.expected(r.item)
        if spacing_ok and U.cos(y, ex_y) >= 0.999:
            ok = True
            break
    yield ok, f"even spacing {a['spacing']} + cos >= 0.999 vs reference"


def P09(r):
    """Merge three files onto a common shift axis and save."""
    a = r.item["answer"]
    got = U.agent_spectra(r)
    yield len(got) == a["n_intensity_columns"], \
        f"{a['n_intensity_columns']} saved files (got {len(got)})"
    yield all(len(y) == a["n_rows"] for _x, y in got), f"{a['n_rows']} rows per file"
    if len(got) >= 2:
        base = got[0][0]
        yield all(U.allclose(x, base) for x, _y in got[1:]), "all files share one shift axis"
    means = sorted(sum(y) / len(y) for _x, y in got if y)
    yield U.set_close(means, sorted(a["column_means"]), tol=1e-3), "column means match the reference"


def P10(r):
    """Savitzky-Golay smoothing (window 11, order 3) and save."""
    a = r.item["answer"]
    yield U.matches_expected(r, r.item)
    yield r.cites_any([11, 3]), "cited parameters (window 11, order 3)"
    got = U.agent_spectra(r)
    if got:
        x, y = got[-1]
        _nx, ny = U.in_range(x, y, 1800, 2000)
        after = (U.stdev_both(ny) or [1e9])[0]
        yield after < a["std_1800_2000_before"], "noise-window stdev lower than the source"


def P11(r):
    """Remove cosmic-ray spikes and report their positions."""
    a = r.item["answer"]
    for pos in a["spike_positions_cm-1"]:
        yield r.cites(pos, tol=0.5), f"cited spike at {pos}"
    yield U.close_enough(r, r.item, min_cos=0.99, max_nrmse=0.02)


def P12(r):
    """Min-max normalization (0-1) and save."""
    yield U.matches_expected(r, r.item)
    got = U.agent_spectra(r)
    yield any(abs(min(y)) < 1e-6 and abs(max(y) - 1.0) < 1e-6 for _x, y in got if y), \
        "result min == 0 and max == 1"


def P13(r):
    """Unit-vector (L2 norm 1) normalization and save."""
    import math
    yield U.matches_expected(r, r.item)
    got = U.agent_spectra(r)
    yield any(abs(math.sqrt(sum(v * v for v in y)) - 1.0) < 1e-6 for _x, y in got if y), \
        "result L2 norm == 1"


def P14(r):
    """Top six peaks by intensity with positions and intensities."""
    top6 = r.item["answer"]["top6_descending"]
    for p in top6:
        yield r.cites(p["shift"], tol=3.0), f"cited peak {p['shift']}"
    yield sum(r.cites(p["intensity"], tol=0.03, rel=True) for p in top6) >= len(top6), \
        "reported all six intensities as well (+-3%)"


def P15(r):
    yield r.cites(r.item["answer"]["fwhm_cm-1"], tol=0.05, rel=True), "cited FWHM (+-5%)"


def P16(r):
    yield r.cites(r.item["answer"]["area_990_1012"], tol=0.02, rel=True), "cited the integrated area (+-2%)"


def P17(r):
    """Extract 800-1200 cm-1 without interpolation and save."""
    a = r.item["answer"]
    yield U.matches_expected(r, r.item)
    got = U.agent_spectra(r)
    yield any(len(y) == a["n_rows"] for _x, y in got), f"{a['n_rows']} rows"


def P18(r):
    """Overlay two spectra on one axis with a legend."""
    a = r.item["answer"]
    yield bool(U.saved_paths(r, "figure")) or r.called("run_analysis"), "produced a plot"
    for name, vals in a.items():
        yield r.cites(vals["y_max"], tol=0.02, rel=True), f"cited y_max of {name} (+-2%)"


def P19(r):
    """Report the peaks recovered after 5th-order polynomial background removal."""
    a = r.item["answer"]
    yield len(a["recovered_peaks_cm-1"]) == a["n_peaks"], "reference peak count (setup check)"
    for pos in a["recovered_peaks_cm-1"]:
        yield r.cites(pos, tol=3.0), f"cited recovered peak {pos}"
    yield U.close_enough(r, r.item, min_cos=0.99, max_nrmse=0.02)


def P20(r):
    """Shift of the sample relative to the reference."""
    yield r.cites(r.item["answer"]["shift_cm-1"], tol=0.2), "cited the shift amount (+-0.2)"
