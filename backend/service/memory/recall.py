"""Query and summary of past measurement records."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field

from backend.common.config import memory as mem_cfg
from backend.common.result import fail, ok
from backend.repo.store import memory_db, run_store, skill_store

_FIELDS = ("record_uid", "created_at", "session_id", "power_mw", "exposure_s",
           "acq_mode", "n_frames", "objective", "x_mm", "y_mm", "z_mm", "sample_name", "sample_source",
           "effective_sample", "expected_bands", "expected_bands_source", "band_values",
           "spectrum_file", "lpips", "error_note", "annotations")


def query_measurements(
    sample_name: Annotated[Optional[str], Field(description="Only records of this sample (case-insensitive). Records whose sample was corrected later by an annotation match their corrected name.")] = None,
    near_x_mm: Annotated[Optional[float], Field(description="Only records measured near this stage X (mm). Use together with near_y_mm.")] = None,
    near_y_mm: Annotated[Optional[float], Field(description="Only records measured near this stage Y (mm). Use together with near_x_mm.")] = None,
    radius_mm: Annotated[Optional[float], Field(gt=0, description="Radius for near_x_mm/near_y_mm in mm. Default 0.005 (5 um).")] = 0.005,
    session: Annotated[Optional[Literal["current", "any"]], Field(description="'any' (default) searches every past session; 'current' only this one.")] = "any",
    limit: Annotated[Optional[int], Field(ge=1, le=50, description="Maximum records, newest first. Default 20.")] = 20,
) -> dict:
    """Query past measurement records.

    Returns:
        dict: ok() with the number of records and the records, or fail().
    """
    near = None
    if near_x_mm is not None and near_y_mm is not None:
        near = (float(near_x_mm), float(near_y_mm), float(radius_mm or 0.005))
    sid = (run_store.current().get("session_id") or None) if session == "current" else None
    rows = memory_db.measurements(sample_name=sample_name, near=near, session_id=sid, limit=int(limit or 20))
    return ok(count=len(rows), records=[{k: r.get(k) for k in _FIELDS} for r in rows],
              note="lpips compares the view right before and right after each shot (larger = the surface "
                   "changed more). It is a raw value with no damage verdict - camera noise, focus and lighting "
                   "also move it, so compare against a control shot before calling a spot damaged. error_note "
                   "means the laser fired but no spectrum came back.")


def annotate_measurement(
    record_id: Annotated[str, Field(description="Id of the record to annotate, as shown by query_measurements or in the past-measurement summary (the first 8 characters are enough).")],
    kind: Annotated[Literal["sample_identity", "quality", "error", "note"], Field(description="'sample_identity' = what the sample turned out to be; 'quality' = a verdict on the spectrum such as 'saturated' or 'signal too weak'; 'error' = what went wrong; 'note' = anything else worth knowing later.")],
    text: Annotated[Optional[str], Field(description="The remark itself. Required for quality, error and note.")] = None,
    sample_name: Annotated[Optional[str], Field(description="Required for sample_identity: what the sample really is.")] = None,
    source: Annotated[Optional[Literal["user", "agent"]], Field(description="'user' if the user told you this, 'agent' if it is your own judgement. Default 'agent'.")] = "agent",
) -> dict:
    """Add an annotation to a measurement record without changing the record.

    Returns:
        dict: ok() with the record id and annotation, or fail().
    """
    try:
        uid = memory_db.resolve_uid(record_id)
    except KeyError as e:
        return fail(str(e).strip("'\""))
    src = source or "agent"
    if kind == "sample_identity":
        if not (sample_name or "").strip():
            return fail("sample_name is required for kind='sample_identity'.")
        body = {"sample_name": sample_name.strip(), "source": src}
    else:
        if not (text or "").strip():
            return fail(f"text is required for kind='{kind}'.")
        body = {"text": text.strip(), "source": src}
    memory_db.insert_annotation(uid, "agent", kind, body)
    note = ("Added. The original record is unchanged; queries now show this as its latest "
            f"'{kind}' annotation.")
    if kind == "sample_identity":
        note += (" The record is now found under both its original sample name and the corrected one, so "
                 "it still appears in queries for the original name.")
    return ok(record_id=uid, kind=kind, annotation=body, note=note)


def summary_text(sample_name: Optional[str] = None, days: float = 30, max_rows: int = 30) -> str:
    """Summarize past records of a sample, one line per measurement.

    Args:
        sample_name (str, optional): Sample to summarize.
        days (float): How many days back to look.
        max_rows (int): Maximum number of lines.

    Returns:
        str: The summary. Empty if there is no sample or no matching record.
    """
    if not sample_name:
        return ""
    rows = memory_db.measurements(sample_name=sample_name, since_days=days, limit=max_rows)
    return "\n".join(_line(r) for r in rows[:max_rows])


def _line(r: dict) -> str:
    def num(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) else "?"
    bands = r.get("band_values") or []
    hit = [b for b in bands if b.get("detected")]
    snrs = [b["snr"] for b in bands if isinstance(b.get("snr"), (int, float))]
    band_txt = f"{len(hit)}/{len(bands)} best_snr={max(snrs):.1f}" if snrs else "-"
    if bands and not _current_judge(r):
        band_txt = "old-judge(unreliable)"
    lp = r.get("lpips")
    return (f"#{r['record_uid'][:8]} {r.get('effective_sample') or '?'}({r.get('sample_source') or '?'}) "
            f"P={num(r.get('power_mw'), '.3g')}mW t={num(r.get('exposure_s'), '.3g')}s{_counts_txt(r)} @({num(r.get('x_mm'), '.4f')},{num(r.get('y_mm'), '.4f')}) "
            f"bands={band_txt} lpips={'-' if lp is None else f'{lp:.3f}'}"
            + (f" ERROR: {r['error_note'][:80]}" if r.get("error_note") else ""))


def _counts(r: dict) -> dict:
    """Return the accumulation and kinetic frame counts of a record."""
    acc, fr = memory_db.acq_counts(r)
    return {"accumulations": acc, "kinetic_frames": fr}


def _counts_txt(r: dict) -> str:
    acc, fr = memory_db.acq_counts(r)
    return (f" x{acc}acc" if acc > 1 else "") + (f" {fr}frames" if r.get("acq_mode") == "kinetic" else "")


RECALL_K, RECENT_N = 20, 30
IDENT_RADIUS_MM = 0.005  # radius for finding identity annotations of the same spot


def spot_hint() -> Optional[dict]:
    """Return the memory hint for the current laser spot.

    Records of the same sample are used first, then skill rules, then recent records when no sample is known.

    Returns:
        dict or None: Best past conditions and earlier shots at this spot, or None.
    """
    if not mem_cfg.RECALL:
        return None
    from backend.service.memory import recorder
    here = _here()
    key = _key(here)
    source, recs, rules = None, [], []
    if mem_cfg.QUERY and key:
        recs = memory_db.measurements(limit=RECALL_K, sample_name=key[1])
        source = "records" if recs else None
    if not recs and mem_cfg.SKILL_INJECT and key:
        rules = _rules_for(key, (recorder.CALL.get() or {}).get("skill_version"))
        ev = sorted({e for r in rules for e in r["evidence"]})
        recs = memory_db.measurements(uids=ev, limit=max(len(ev), 1)) if ev else []
        source = "skill" if rules else None
    if not recs and not rules and mem_cfg.SUMMARY and not key:
        recs = memory_db.measurements(limit=RECENT_N)
        source = "recent" if recs else None
    if source is None:
        return None
    out = {"key": f"{key[0]}:{key[1]}" if key else None, "source": source, "n_records": len(recs)}
    if here.get("spot_so_far"):
        out["spot_so_far"] = here["spot_so_far"]
    best = _best(recs)
    if best:
        out["best_past"] = best
    if rules:
        out["rules"] = [r["rule"] for r in rules]
    out["note"] = _NOTE[source]
    return out


_NOTE = {
    "records": "Past measurements of this sample (the declared sample, or the material identified at this "
               "spot). best_past is the condition that gave the best expected-band SNR. spot_so_far lists "
               "every shot already fired at this exact spot in this session (power, exposure, frames), "
               "oldest first. Learned rules are not shown when past "
               "records exist; read them with open_file('memory:latest.md') if needed. Advisory - you decide.",
    "skill": "No past records of this sample; these are learned rules for it and the measurements behind "
             "them. Advisory.",
    "recent": "The sample here is not known (none declared or identified); this comes from the most recent measurements of ANY "
              "sample, so it is only a rough prior. Advisory.",
}


def _here() -> dict:
    """Return the current position, identified material, shot history and declared sample."""
    from backend.repo.hw_access import hw_core as _hw
    from backend.service.memory import recorder
    out = {"declared": recorder.sample_ctx().get("sample_name")}
    try:
        pos = _hw._stage.get_position() if _hw._stage is not None else None
        if pos is not None:
            out["x"], out["y"] = float(pos[0]), float(pos[1])
    except Exception:
        pass
    if "x" in out:
        sid = run_store.current().get("session_id") or None
        near = memory_db.measurements(near=(out["x"], out["y"], recorder.SPOT_RADIUS_MM), session_id=sid, limit=50)
        if near:
            out["spot_so_far"] = [{"power_mw": r.get("power_mw"), "exposure_s": r.get("exposure_s"),
                                   **_counts(r)} for r in reversed(near)]
        ident = [memory_db.identity_of(r) for r in
                 memory_db.measurements(near=(out["x"], out["y"], IDENT_RADIUS_MM), session_id=sid, limit=10)]
        out["identity"] = next((i for i in ident if i), None)
    return out


def _key(here: dict):
    name = here.get("identity") or here.get("declared")
    return ("sample", name) if name else None


def _rules_for(key, version) -> list[dict]:
    """Return the pinned skill rules whose scope matches the sample."""
    if not version:
        return []
    _, rules = skill_store.read(int(version))
    want = str(key[1]).casefold()
    return [skill_store.canonical(r) for r in rules
            if isinstance(r.get("scope"), dict) and str(r["scope"].get(key[0], "")).casefold() == want
            and r.get("status") != "candidate"]


def _current_judge(r: dict) -> bool:
    """Return whether a record's band_values were computed with the current judge."""
    from backend.common import raman_judge
    return (r.get("spectrum_qc") or {}).get("judge_version") == raman_judge.JUDGE_VERSION


def _best(recs: list[dict]) -> Optional[dict]:
    """Return the record with the best expected-band detection."""
    def snr(r):
        if not _current_judge(r):
            return None
        v = [b["snr"] for b in (r.get("band_values") or [])
             if b.get("detected") and isinstance(b.get("snr"), (int, float))]
        return max(v) if v else None
    ok_ = [(snr(r), r) for r in recs if snr(r) is not None]
    if not ok_:
        return None
    s_, r = max(ok_, key=lambda x: x[0])
    return {"record_id": r["record_uid"][:8], "power_mw": r.get("power_mw"), "exposure_s": r.get("exposure_s"),
            **_counts(r), "best_snr": s_, "lpips": r.get("lpips"),
            "material": memory_db.identity_of(r)}


BAND_DROP_RATIO = 0.8  # band ratio below which a band is reported as decreased
BAND_CHANGE_SIGMA = 3.0  # minimum change in noise sigma
SI_RISE_RATIO = 1.3  # Si ratio above which the sample is reported as thinner or gone
_CRITERIA = (f"spectral_change is true if a target band fell below {BAND_DROP_RATIO}x by >= {BAND_CHANGE_SIGMA} "
             f"sigma (a band that merely crossed the detection threshold without a {BAND_CHANGE_SIGMA}-sigma change is "
             f"listed under 'threshold_flips', not counted), a marker band appeared with a >= {BAND_CHANGE_SIGMA}-sigma "
             f"rise, a new peak appeared (SNR >= 8 where the before probe had < 4), or Si grew > {SI_RISE_RATIO}x. "
             f"lpips is reported raw: compare it with the lpips of two probes on an UNirradiated spot before calling "
             f"the view changed.")


def _change_reasons(cmp_: dict) -> tuple[list[str], list[str]]:
    """Return (reasons for change, insignificant detection flips) from compare_spectra output."""
    reasons, flips = [], []
    for r in cmp_["bands"]:
        sig = r.get("change_sigma")
        if r["role"] == "target" and r.get("ratio") is not None and r["ratio"] < BAND_DROP_RATIO \
                and (sig or 0) <= -BAND_CHANGE_SIGMA:
            reasons.append(f"target band {r['center_cm1']:g} fell to {r['ratio']:.2f}x ({sig:.1f} sigma)")
        elif r["role"] == "target" and r["before_detected"] and r["after_detected"] is False:
            if sig is not None and sig <= -BAND_CHANGE_SIGMA:
                reasons.append(f"target band {r['center_cm1']:g} vanished ({sig:.1f} sigma)")
                continue
            flips.append(f"target band {r['center_cm1']:g} detected before, not after, but changed only "
                         f"{sig if sig is not None else '?'} sigma")
        if r["role"] == "marker" and r["after_detected"] and not r["before_detected"]:
            if sig is not None and sig >= BAND_CHANGE_SIGMA:
                reasons.append(f"marker band {r['center_cm1']:g} appeared (+{sig:.1f} sigma)")
            else:
                flips.append(f"marker band {r['center_cm1']:g} crossed the detection threshold, but rose only "
                             f"{sig if sig is not None else '?'} sigma")
    if cmp_["new_peaks"]:
        reasons.append("new peaks at " + ", ".join(f"{p['at_cm1']:g}" for p in cmp_["new_peaks"]))
    if cmp_["si_ratio"] and cmp_["si_ratio"] > SI_RISE_RATIO:
        reasons.append(f"Si 520 grew {cmp_['si_ratio']:.2f}x (sample thinner or gone)")
    return reasons, flips


def _load_spectrum(rel: Optional[str]):
    """Load (axis, intensity) of a record's spectrum file."""
    import numpy as np
    from backend.repo.store import DATA_ROOT
    if not rel:
        return None, None
    p = DATA_ROOT / rel
    if not p.exists():
        return None, None
    xs, ys = [], []
    with open(p, encoding="utf-8-sig") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split(",")
            if header is None:
                header = cols
                continue
            if "raman_shift_cm-1" not in header or "intensity" not in header:
                return None, None
            xs.append(float(cols[header.index("raman_shift_cm-1")]))
            ys.append(float(cols[header.index("intensity")]))
    return (np.asarray(xs), np.asarray(ys)) if xs else (None, None)


def compare_measurements(
    before_record_id: Annotated[str, Field(description="Record id (>= 8 chars) of the low-power probe taken at the spot BEFORE the irradiation you are testing.")],
    after_record_id: Annotated[Optional[str], Field(description="Record id of the probe taken at the SAME spot AFTER that irradiation (same power/exposure/frames as the before probe, ideally after a wait so reversible effects can recover). OMIT it to check a single KINETIC measurement against itself: its first third of frames is compared with its last third, which shows whether the spot changed during that one acquisition without spending extra shots.")] = None,
) -> dict:
    """Compare probe measurements taken at the same spot before and after irradiation.

    Returns:
        dict: ok() with band changes, new peaks, silicon ratio and LPIPS, or fail().
    """
    import math

    from backend.common import raman_judge
    from backend.service.memory import recorder
    if not after_record_id:
        return _compare_within(before_record_id)
    try:
        ua, ub = memory_db.resolve_uid(before_record_id), memory_db.resolve_uid(after_record_id)
    except KeyError as e:
        return fail(str(e).strip("'\""))
    rows = {r["record_uid"]: r for r in memory_db.measurements(uids={ua, ub}, limit=2)}
    a, b = rows.get(ua), rows.get(ub)
    if not a or not b:
        return fail("record not found")
    xa, ya = _load_spectrum(a.get("spectrum_file"))
    xb, yb = _load_spectrum(b.get("spectrum_file"))
    if xa is None or xb is None:
        return fail("One of the two records has no calibrated single/accumulate spectrum file to compare.")
    bands = list(a.get("expected_bands") or []) + list(a.get("marker_bands") or [])
    if not bands:
        return fail("The before record carries no expected/marker bands - declare them with set_sample_context "
                    "(sourced) before probing.")
    warnings = []
    sig_scale = lambda r: (r.get("power_mw") or 0) * (r.get("exposure_s") or 0) * (r.get("n_frames") or 1)
    scale = sig_scale(a) / sig_scale(b) if sig_scale(a) and sig_scale(b) else 1.0
    if abs(scale - 1.0) > 0.01:
        warnings.append(f"The two probes used different power x exposure x frames; the after spectrum was scaled by "
                        f"{scale:.3g} (assumes signal is linear in each). Prefer identical probe conditions.")
    if None not in (a.get("x_mm"), a.get("y_mm"), b.get("x_mm"), b.get("y_mm")):
        d_um = 1000 * math.hypot(a["x_mm"] - b["x_mm"], a["y_mm"] - b["y_mm"])
        if d_um > 1000 * recorder.SPOT_RADIUS_MM:
            warnings.append(f"The two probes are {d_um:.1f} um apart - not the same spot.")
    cmp_ = raman_judge.compare_spectra(xa, ya, xb, yb, bands, scale_b=scale)
    lp = None
    try:
        import cv2
        from backend.repo.store import DATA_ROOT
        ia, ib = a.get("image_before_file"), b.get("image_after_file")
        if ia and ib and (DATA_ROOT / ia).exists() and (DATA_ROOT / ib).exists():
            lp = recorder.lpips_distance(cv2.imread(str(DATA_ROOT / ia)), cv2.imread(str(DATA_ROOT / ib)))
    except Exception:
        lp = None
    reasons, flips = _change_reasons(cmp_)
    return ok(before=ua, after=ub, spectral_change=bool(reasons), reasons=reasons, threshold_flips=flips, lpips_before_vs_after=lp,
              bands=cmp_["bands"], new_peaks=cmp_["new_peaks"], si_ratio=cmp_["si_ratio"],
              before_qc=cmp_["before_qc"], after_qc=cmp_["after_qc"], warnings=warnings,
              criteria=_CRITERIA)


def _load_frames(rel: Optional[str]):
    """Load (axis, frames) of a kinetic record."""
    import json as _json

    import numpy as np
    from backend.repo.store import DATA_ROOT
    if not rel:
        return None, None
    j = (DATA_ROOT / rel).with_suffix(".json")
    if not j.exists():
        return None, None
    try:
        frames = (_json.loads(j.read_text(encoding="utf-8")).get("result") or {}).get("frames") or []
        if len(frames) < 2 or "raman_shift_cm-1" not in frames[0]:
            return None, None
        return np.asarray(frames[0]["raman_shift_cm-1"], float), np.asarray([f["intensity"] for f in frames], float)
    except Exception:
        return None, None


def _compare_within(record_id: str) -> dict:
    """Compare the first and last third of the frames of one kinetic record."""
    from backend.common import raman_judge
    try:
        uid = memory_db.resolve_uid(record_id)
    except KeyError as e:
        return fail(str(e).strip("'\""))
    r = (memory_db.measurements(uids={uid}, limit=1) or [None])[0]
    if not r:
        return fail("record not found")
    x, fr = _load_frames(r.get("spectrum_file"))
    if x is None or len(fr) < 3:
        return fail("A single-record check needs a kinetic measurement with at least 3 frames (acq_mode='kinetic', "
                    "kinetic_count >= 3). Otherwise give after_record_id.")
    bands = list(r.get("expected_bands") or []) + list(r.get("marker_bands") or [])
    if not bands:
        return fail("The record carries no expected/marker bands - declare them with set_sample_context first.")
    k = max(1, len(fr) // 3)
    cmp_ = raman_judge.compare_spectra(x, fr[:k].mean(axis=0), x, fr[-k:].mean(axis=0), bands)
    reasons, flips = _change_reasons(cmp_)
    return ok(record=uid, mode="within-kinetic", frames_compared=f"first {k} vs last {k} of {len(fr)}",
              spectral_change=bool(reasons), reasons=reasons, threshold_flips=flips,
              lpips_before_vs_after=r.get("lpips"),
              bands=cmp_["bands"], new_peaks=cmp_["new_peaks"], si_ratio=cmp_["si_ratio"],
              criteria=_CRITERIA)
