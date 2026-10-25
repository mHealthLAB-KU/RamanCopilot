"""Writes a new skill version from the records of a finished run."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable

import yaml

from backend.repo.store import memory_db, skill_store

SYSTEM = """You maintain the skill memory of an autonomous Raman microscope agent.
Skills are short, general lessons for future tasks. The agent can read the current rules at any time with
open_file('memory:latest.md'); when it moves to or measures a spot, an ACTIVE rule is also shown if its scope
matches the sample there AND no past measurement record of that sample exists (records are shown first).

You receive (1) the measurement records of the task that just finished and (2) the skill rules that
are active right now. Write the COMPLETE set of rules that should be active from now on - a full
snapshot, not a diff.

Each rule has exactly three keys:
  rule:     one sentence, a lesson that changes how a future measurement should be done
  scope:    exactly {sample: <sample name>}
  evidence: list of record ids that support the rule

How the records were judged (so you do not learn the judge's blind spots as facts):
  - band_values were computed by the current judge from each saved spectrum: the Si reference spectrum is
    subtracted and the axis is shifted onto Si 520.7 first. detected=null means NOT JUDGED (the window touches
    the Si line, or the spectrum is saturated) - it is not evidence that the band is absent.
  - A band can only be called absent if it was judged (detected=false) with a width (fwhm_cm1) that fits the
    band. Never write that a band is "not resolvable" or "absent" from records where it was not judged.
  - spectrum_qc gives si_height (substrate signal) and background_counts (fluorescence). A strong signal with
    no detected expected band is fluorescence or substrate, not the sample.

Hard requirements (rules that break them are discarded automatically):
  - Every rule must have at least 3 evidence records, from at least 2 different spots (different spot_label or
    x/y more than 2 um apart). One spot is an anecdote, not a lesson.
  - Every evidence id must be a record id that appears in the input.
  - scope {sample: S}: every evidence record's effective_sample is S (case-insensitive).
    Any other scope ({all: true}, a free description, ...) is discarded.
  - If the rule names a band position (a number in cm-1) and says it is detected/strong/present, an evidence
    record must show a detected band within 30 cm-1 of it; if it says absent/weak/not detected/not resolvable,
    no evidence record may show that band detected. Contradicted rules are discarded.
  - lpips is a raw before/after image distance with no damage verdict; camera noise, focus and lighting
    move it too. Do not call a spot damaged from one lpips value alone.
  - A rule supported by records of only one task is stored as a CANDIDATE and is not shown to the agent
    until records from a later task also support it. So do not overstate a one-task finding.

Also:
  - Generalise only as far as the evidence reaches.
  - Keep the current rules unchanged unless the new records contradict or refine them.
  - If this task taught nothing new, output the current rules exactly as they are.
  - Do not invent numbers that are not in the records.

Output YAML only: a list of rules (an empty list if there are none). No prose, no code fences."""


def reflect(run_id: str, ask: Callable[[str, str], str], model: str = "") -> dict:
    """Run reflection on a finished run and write a new skill version if the rules changed.

    Args:
        run_id (str): Run to reflect on.
        ask (Callable[[str, str], str]): Function that sends a system and user prompt to the LLM.
        model (str): Model name stored in the skill file.

    Returns:
        dict: New version number or None, reason and dropped rules.
    """
    recs = [rejudge(r) for r in memory_db.measurements(run_id=run_id, limit=500)]
    if not recs:
        return {"version": None, "reason": "no measurements in this run", "dropped": []}
    cur_v = skill_store.latest_version()
    _, cur_rules = skill_store.read(cur_v)
    cur_rules = [skill_store.canonical(r) for r in cur_rules]
    allowed = {r["record_uid"] for r in recs} | {e for r in cur_rules for e in r["evidence"]}

    out = _parse(ask(SYSTEM, _prompt(recs, cur_rules)))
    if out is None:
        out = _parse(ask(SYSTEM, _prompt(recs, cur_rules) + _RETRY_NOTE))
    if out is None:
        return {"version": None, "reason": "reflection output was empty or could not be parsed (twice)", "dropped": []}

    touched = _touched(cur_rules, out)
    first_wipes = out == []
    again = None
    if touched:
        prior_ids = {e for r in touched for e in r["evidence"]} - {r["record_uid"] for r in recs}
        prior = [rejudge(r) for r in memory_db.measurements(uids=prior_ids, limit=len(prior_ids))] if prior_ids else []
        allowed |= {r["record_uid"] for r in prior}
        again = _parse(ask(SYSTEM, _prompt(recs, cur_rules, prior=prior, draft=out)))
        if again is not None:
            out = again

    if cur_rules and out == [] and not (first_wipes and again == []):
        return {"version": None, "reason": "wiping every rule was not confirmed by the second pass", "dropped": []}

    rules, dropped = validate(out, allowed, recs_by_uid={r["record_uid"]: r for r in recs})
    if out and not rules:
        return {"version": None, "reason": "every proposed rule failed validation", "dropped": dropped}
    if skill_store.same(rules, cur_rules):
        return {"version": None, "reason": "no change", "dropped": dropped}
    v = skill_store.write(rules, {"run_id": run_id, "model": model,
                                  "created_at": datetime.now().isoformat(timespec="seconds")})
    return {"version": v, "reason": "new version", "dropped": dropped}


def validate(rules: list, allowed: set, recs_by_uid: dict | None = None) -> tuple[list[dict], list]:
    """Check proposed rules against their evidence.

    Args:
        rules (list): Rules written by the LLM.
        allowed (set): Record ids that may be cited as evidence.
        recs_by_uid (dict, optional): Records by id.

    Returns:
        tuple[list[dict], list]: Accepted rules, and rejected rules with the reason.
    """
    ev_ids = {str(e) for r in rules if isinstance(r, dict) for e in (r.get("evidence") or [])} & allowed
    known = dict(recs_by_uid or {})
    miss = ev_ids - set(known)
    if miss:
        known.update({r["record_uid"]: rejudge(r) for r in memory_db.measurements(uids=miss, limit=len(miss))})
    kept, dropped = [], []
    for raw in rules:
        r = skill_store.canonical(raw)
        if not r["rule"]:
            dropped.append((r, "empty rule")); continue
        if not r["evidence"]:
            dropped.append((r, "no evidence")); continue
        missing = [e for e in r["evidence"] if e not in allowed]
        if missing:
            dropped.append((r, f"evidence not in input: {missing}")); continue
        ev = [known[e] for e in r["evidence"] if e in known]
        why = _check(r, ev) or _check_support(r, ev)
        if why:
            dropped.append((r, why)); continue
        runs = {e.get("run_id") for e in ev if e.get("run_id")}
        r["status"] = "active" if len(runs) >= MIN_RUNS_ACTIVE else "candidate"
        kept.append(r)
    return kept, dropped


def _check(rule: dict, ev: list[dict]) -> str | None:
    """Check scope and evidence of a rule; return a rejection reason or None."""
    sc = rule["scope"]
    if not (isinstance(sc, dict) and len(sc) == 1 and next(iter(sc)) == "sample"
            and str(sc["sample"] or "").strip()):
        return "scope must be exactly {sample: <sample name>}"
    val = str(sc["sample"]).strip()
    if any(str(r.get("effective_sample") or "").casefold() != val.casefold() for r in ev):
        return f"not every evidence record is sample '{val}'"
    return None


MIN_EVIDENCE = 3  # records required to support a rule
MIN_SPOTS = 2  # distinct spots required to support a rule
MIN_RUNS_ACTIVE = 2  # distinct runs required before a rule is used
CLAIM_TOL_CM1 = 30.0  # tolerance between band numbers in a rule and its evidence
_JUDGE_TALK = re.compile(r"prominence|peak_inside|peak inside|detection window|expected (peak )?window", re.I)
_NEG = re.compile(r"\b(not|no|never|absent|weak|undetect\w*|unresolv\w*|miss\w*|lack\w*|cannot|can't|without)\b",
                  re.I)


def _spots(ev: list[dict]) -> int:
    """Return the number of distinct spots among the evidence records."""
    from backend.service.memory.recorder import SPOT_RADIUS_MM
    labels = {e["spot_label"] for e in ev if e.get("spot_label")}
    pts = []
    for e in ev:
        if e.get("spot_label") or e.get("x_mm") is None or e.get("y_mm") is None:
            continue
        if not any(abs(e["x_mm"] - x) <= SPOT_RADIUS_MM and abs(e["y_mm"] - y) <= SPOT_RADIUS_MM for x, y in pts):
            pts.append((e["x_mm"], e["y_mm"]))
    return len(labels) + len(pts)


def _check_support(rule: dict, ev: list[dict]) -> str | None:
    """Check that the evidence actually supports the rule."""
    if len(ev) < MIN_EVIDENCE:
        return f"needs >= {MIN_EVIDENCE} evidence records (has {len(ev)})"
    if _spots(ev) < MIN_SPOTS:
        return f"evidence comes from {_spots(ev)} spot(s); needs >= {MIN_SPOTS} different spots"
    text = rule["rule"]
    if _JUDGE_TALK.search(text):
        return ("the rule is about the judge's internal numbers (prominence_snr / peak_inside), not about the sample "
                "or the conditions - those numbers changed meaning when the judge was replaced")
    nums = [float(m) for m in re.findall(r"(?<![\d.])(\d{3,4}(?:\.\d+)?)(?![\d.])", text)]
    nums = [n for n in nums if 150 <= n <= 3500]
    if not nums:
        return None
    negative = bool(_NEG.search(text))
    for n in nums:
        near = [b for e in ev for b in (e.get("band_values") or [])
                if isinstance(b.get("center_cm1"), (int, float)) and abs(b["center_cm1"] - n) <= CLAIM_TOL_CM1]
        declared = any(abs(float(b.get("center_cm1", -1e9)) - n) <= CLAIM_TOL_CM1
                       for e in ev for b in (e.get("expected_bands") or []) + (e.get("marker_bands") or []))
        if not near:
            if declared:
                return (f"the rule is about the band near {n:g}, but none of its evidence spectra can be re-judged "
                        f"with the current judge (spectrum file missing), so it cannot be verified")
            continue
        hit = any(b.get("detected") for b in near)
        judged = any(b.get("detected") is not None for b in near)
        if not judged:
            return (f"the band near {n:g} was never judged in the evidence (it overlaps the Si 520 line or the "
                    f"spectra are saturated), so nothing can be concluded about it")
        if negative and hit:
            return f"says the band near {n:g} is absent/weak, but an evidence record shows it detected"
        peaks = [p["at_cm1"] for e in ev for p in (e.get("residual_peaks") or []) if abs(p["at_cm1"] - n) <= CLAIM_TOL_CM1]
        if negative and peaks:
            return (f"says the band near {n:g} is absent/weak, but the evidence spectra show a peak at "
                    f"{sorted(peaks)[len(peaks) // 2]:g} (the declared window probably missed it)")
        if not negative and not hit and not peaks:
            return f"says the band near {n:g} is present, but no evidence record shows it detected"
    return None


def rejudge(r: dict) -> dict:
    """Recompute the band values of a record from its stored spectrum.

    Args:
        r (dict): Measurement record.

    Returns:
        dict: Record with updated band values, or with the old values removed if the spectrum cannot be read.
    """
    from backend.common import raman_judge
    from backend.service.memory import recall
    qc = r.get("spectrum_qc") or {}
    bands = list(r.get("expected_bands") or []) + list(r.get("marker_bands") or [])
    out = dict(r)
    try:
        x, y = recall._load_spectrum(r.get("spectrum_file"))
    except Exception:
        x, y = None, None
    if x is not None:
        out["residual_peaks"] = raman_judge.residual_peaks(x, y)
    if qc.get("judge_version") == raman_judge.JUDGE_VERSION:
        return out
    if x is None:
        if out.get("band_values"):
            out["band_values"] = None
            out["band_values_note"] = "dropped: computed by the retired judge and the spectrum file is unavailable"
        return out
    out["spectrum_qc"] = raman_judge.spectrum_qc(x, y)
    out["band_values"] = raman_judge.judge_bands(x, y, bands) if bands else None
    return out


def _touched(cur: list[dict], new: list) -> list[dict]:
    """Return existing rules that are modified or removed in the new snapshot."""
    key = lambda r: (r["rule"], json.dumps(r["scope"], sort_keys=True, ensure_ascii=False))
    keep = {key(skill_store.canonical(r)) for r in new}
    return [r for r in cur if key(r) not in keep]


def _brief(r: dict) -> dict:
    keys = ("record_uid", "run_id", "effective_sample", "sample_source", "power_mw", "exposure_s",
            "acq_mode", "x_mm", "y_mm", "spot_label", "purpose", "expected_bands_source", "band_values",
            "band_values_note", "lpips", "error_note")
    out = {k: r.get(k) for k in keys if r.get(k) is not None}
    out["accumulations"], out["kinetic_frames"] = memory_db.acq_counts(r)
    qc = r.get("spectrum_qc") or {}
    if qc:
        out["spectrum_qc"] = {k: qc.get(k) for k in ("saturated", "si_height", "axis_shift_cm1", "noise_sigma",
                                                    "background_counts")}
    if memory_db.identity_of(r):
        out["identity"] = memory_db.identity_of(r)
    notes = {k: v for k, v in (r.get("annotations") or {}).items() if k != "sample_identity"}
    if notes:
        out["annotations"] = notes
    return out


def _prompt(recs, cur_rules, prior=None, draft=None) -> str:
    parts = ["## Records of the task that just finished",
             json.dumps([_brief(r) for r in recs], ensure_ascii=False, indent=1),
             "## Current skill rules",
             yaml.safe_dump(cur_rules, allow_unicode=True, sort_keys=False) if cur_rules else "(none)"]
    if prior is not None:
        parts += ["## Your draft changes existing rules. These are the records those rules were based on;"
                  " weigh both sets of evidence before you rewrite or drop a rule.",
                  json.dumps([_brief(r) for r in prior], ensure_ascii=False, indent=1),
                  "## Your draft", yaml.safe_dump(draft, allow_unicode=True, sort_keys=False)]
    parts.append("Lessons are about lpips (raw before/after image distance; larger = the view changed more, no "
                 "damage verdict), band_values (did the "
                 "expected peaks appear, with what SNR) and error_note, against power, exposure and frames. "
                 "Treat these three separately: the same power x time can act differently. "
                 "Records at the same x_mm/y_mm are repeated shots on one spot. "
                 "Scope by sample (effective_sample).")
    return "\n\n".join(parts)


_RETRY_NOTE = ("\n\nYour previous answer could not be read as YAML (it was empty or cut off before the end). "
               "Answer again with the YAML list only, complete, and keep each rule to at most 8 evidence ids.")


def _parse(text: str):
    """Parse the YAML rule list; None if unreadable or empty."""
    if not isinstance(text, str) or not text.strip():
        return None
    fenced = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", text, flags=re.S)
    t = fenced.group(1) if fenced else text.strip()
    try:
        data = yaml.safe_load(t)
    except yaml.YAMLError:
        return None
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        data = data["rules"]
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        return None
    return data
