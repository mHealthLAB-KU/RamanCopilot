
from __future__ import annotations

_MATERIALS = ("PET", "PMMA", "polystyrene", "calcite", "aragonite", "silicon")

_ORGANIC = {"PET", "PMMA", "polystyrene"}


def _only(r, *want: str):
    """Return whether the answer names want and none of the other materials."""
    others = [m for m in _MATERIALS if m not in want]
    named = [m for m in others if r.mentions(m)]
    return not named, f"no incorrect material named (found {named})" if named else "no incorrect material named"


def L01(r):
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield _only(r, a["material"])


def L02(r):
    """Top three matches with rank and score, in the correct order."""
    top3 = r.item["answer"]["top3"]
    names = [t["material"] for t in top3]
    text = (r.answer or "").lower()
    pos = [text.find(n.lower()) for n in names]
    yield all(p >= 0 for p in pos), f"reported all three materials ({names})"
    yield all(p >= 0 for p in pos) and pos == sorted(pos), "mention order matches the ranking"
    yield all(r.cites(t["score"], tol=0.09) for t in top3), "cited all three scores (+-0.09)"


def L03(r):
    """Best similarity below 0.75: no reliable match; naming a material is wrong."""
    yield r.mentions("no reliable", "reliable", "none", "no match", "not in the library",
                     "not in library"), "judged as no reliable match"
    named = [m for m in _MATERIALS if r.mentions(m)]
    yield not named, f"did not commit to a specific material (found {named})"


def L04(r):
    """Two-component mixture: dominant component and similarity of both candidates."""
    a = r.item["answer"]
    yield r.mentions(a["dominant"]), f"dominant component {a['dominant']}"
    yield all(r.mentions(m) for m in a["candidates"]), f"offered candidates {a['candidates']}"
    yield all(r.cites(s, tol=0.06) for s in a["scores"]), "cited both similarities (+-0.06)"


def L05(r):
    """Calcite or aragonite."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"judged as {a['material']}"
    yield not r.claims_negative(a["material"]), f"did not negate {a['material']}"


def L06(r):
    """Single peak near 520 cm-1: silicon, with the peak position."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield r.cites(a["basis_peak_cm-1"], tol=3.0) or r.cites(a["library_peak_cm-1"], tol=3.0), \
        "cited the basis peak position (+-3 cm-1)"


def L07(r):
    """At least two peaks that distinguish the best match from the runner-up."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    hits = [p for p in a["acceptable_distinguishing_peaks"] if r.cites(p, tol=3.0)]
    yield len(hits) >= 2, f"at least 2 distinguishing peaks (got {hits})"


def L08(r):
    """Identification plus organic polymer / inorganic mineral classification."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    organic = a["material"] in _ORGANIC
    yield r.mentions("organic", "polymer") is organic, \
        f"class == {'organic polymer' if organic else 'inorganic mineral'}"


def L09(r):
    """Weak signal: Savitzky-Golay smoothing (window 11, order 3), then identification."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield r.cites(11) and r.cites(3), "cited smoothing parameters (window 11, order 3)"


def L10(r):
    """Whole spectrum shifted by an unknown constant: correct, identify and report the shift."""
    a = r.item["answer"]
    yield r.cites(a["shift_cm-1"], tol=0.2), f"cited the shift {a['shift_cm-1']} (+-0.2)"
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield _only(r, a["material"])


def L11(r):
    """Raw data: background correction and normalization, then identification."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield _only(r, a["material"])


def L12(r):
    """Cite the exact spectrum_id."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield r.mentions(a["spectrum_id"]), f"cited spectrum_id {a['spectrum_id']}"


def L13(r):
    """Top-3 for each of five spectra; the two unregistered ones must be reported as not in the library."""
    a = r.item["answer"]
    for fname, exp in a.items():
        verdict = str(exp.get("verdict") or "")
        if "absent" in verdict.lower() or "not in" in verdict.lower():
            yield r.mentions(fname.removesuffix(".csv")), f"mentioned {fname}"
        else:
            names = [t["material"] for t in exp.get("top3") or []]
            yield all(r.mentions(n) for n in names[:1]), f"top match of {fname} is {names[:1]}"
    yield r.mentions("not in the library", "not in library", "no reliable", "unregistered"), \
        "reported the 2 unregistered queries as not in the library"


def L14(r):
    """Noisy spectrum: identify and flag as borderline if the top-1 similarity is below 0.85."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']}"
    yield r.cites(a["top1_score"], tol=0.05), f"cited top1 similarity {a['top1_score']} (+-0.05)"
    yield r.mentions("marginal", "borderline", "low confidence"), \
        "flagged the match as marginal"


def L15(r):
    """Whether the top match changes before and after background removal."""
    a = r.item["answer"]
    yield r.mentions(a["top1_before"]), f"top match before removal: {a['top1_before']}"
    yield r.mentions(a["top1_after"]), f"top match after removal: {a['top1_after']}"
    yield r.mentions("change", "differ", "switch"), "reported that the top match changed"


def L16(r):
    """Use only the 400-1800 cm-1 range."""
    a = r.item["answer"]
    yield r.mentions(a["material"]), f"identified as {a['material']} (result of the window restriction)"
    yield r.cites(400, tol=1e-6) and r.cites(1800, tol=1e-6), "cited the 400-1800 cm-1 window"


def L17(r):
    """The two most similar materials within the library."""
    a = r.item["answer"]
    yield all(r.mentions(m) for m in a["pair"]), f"reported the pair {a['pair']}"
    yield r.cites(a["similarity"], tol=0.05), f"cited similarity {a['similarity']} (+-0.05)"


def L18(r):
    """Library entries with a peak near 1001 cm-1."""
    a = r.item["answer"]
    want = a["spectrum_ids"]
    yield all(r.mentions(s) for s in want), f"reported {want}"
    others = [f"REF0{i}" for i in range(1, 7)]
    extra = [s for s in others if r.mentions(s) and not any(s in w for w in want)]
    yield not extra, f"no out-of-answer entries named (found {extra})"


def L19(r):
    """Two-material mixture: components and ratio."""
    a = r.item["answer"]
    yield all(r.mentions(m) for m in a["components"]), f"components {a['components']}"
    yield all(r.cites(v, tol=0.05) for v in a["true_ratio"].values()), "cited the ratios (+-0.05)"


def L20(r):
    """Euclidean-distance matching before and after normalization."""
    a = r.item["answer"]
    yield r.mentions(a["top1_raw"]), f"top match before normalisation: {a['top1_raw']}"
    yield r.mentions(a["top1_normalized"]), f"top match after normalisation: {a['top1_normalized']}"
    yield r.mentions("differ", "not the same", "changed", "different"), \
        "reported that the two results differ"
