
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOG_DIR = _ROOT / "DetailLog"

_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


_INV_CM_RE = re.compile(r"\bcm\s*\}?\s*(?:\^|\*\*)?\s*\{?\s*[-−]\s*1\s*\}?", re.I)

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

PEAK_RANGE = (100.0, 4000.0)

_SEGMENT_RE = re.compile(r"[\n.;,]+")

_NEGATIVE = ("not connected", "disconnected", "unavailable", "not available", "offline",
             "no connection", "not detected", "not initialized", "missing", "down",
             "disconnect", "absent")


def read_turn(arch: str, sid: str) -> dict:
    """Read this item's turn entry from the detail log (empty dict if missing)."""
    safe = re.sub(r"[^0-9A-Za-z_-]", "-", sid)[:64] or "nosession"
    hits = sorted(LOG_DIR.glob(f"{arch}_*_{safe}.json"), key=lambda p: p.stat().st_mtime)
    if not hits:
        return {}
    doc = json.loads(hits[-1].read_text(encoding="utf-8"))
    turns = doc.get("turns") or []
    return turns[-1] if turns else {}


def _dig(obj, path: str):
    """Look up a nested dict with dotted keys such as 'ccd.shutter_mode'."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


@dataclass
class Run:
    """All results of running one benchmark item."""

    item: dict
    sid: str
    before: dict
    after: dict
    answer: str
    turn: dict
    files_before: set = field(default_factory=set)
    files_after: set = field(default_factory=set)
    setup: dict = field(default_factory=dict)


    @property
    def calls(self) -> list[dict]:
        """Return the tool call records in order."""
        return list(self.turn.get("Tool calls") or [])

    @property
    def order(self) -> list[str]:
        """Return the names of called tools in order."""
        return list(self.turn.get("tool_call_order") or [])

    def called(self, name: str) -> bool:
        return any(c.get("name") == name for c in self.calls)

    def n_called(self, name: str) -> int:
        return sum(1 for c in self.calls if c.get("name") == name)

    def not_called(self, *names: str) -> bool:
        """Return whether none of the given tools was called."""
        return not any(c.get("name") in names for c in self.calls)

    def sequence(self, first: str, second: str) -> bool:
        """Return whether second was called at some point after first."""
        names = [c.get("name") for c in self.calls]
        if first not in names or second not in names:
            return False
        return names.index(first) < len(names) - 1 - names[::-1].index(second)

    def args(self, name: str, i: int = 0) -> dict:
        """Return the arguments of the i-th call of a tool (empty dict if none)."""
        hits = [c for c in self.calls if c.get("name") == name]
        return dict(hits[i].get("args") or {}) if len(hits) > i else {}

    def all_args(self, name: str) -> list[dict]:
        return [dict(c.get("args") or {}) for c in self.calls if c.get("name") == name]

    def result(self, name: str, i: int = 0) -> dict:
        hits = [c for c in self.calls if c.get("name") == name]
        return dict(hits[i].get("result") or {}) if len(hits) > i else {}

    def ok(self, name: str, i: int = 0) -> bool:
        hits = [c for c in self.calls if c.get("name") == name]
        return bool(hits[i].get("ok")) if len(hits) > i else False

    def failed(self, name: str) -> bool:
        """Return whether the tool was called and refused."""
        return any(c.get("name") == name and not c.get("ok") for c in self.calls)

    def unknown_tools(self) -> list[str]:
        """Return calls to tool names not in the schema."""
        out = []
        for c in self.calls:
            err = str((c.get("result") or {}).get("error") or c.get("error") or "")
            if err.startswith("Unknown tool"):
                out.append(c.get("name"))
        return out

    @property
    def dose(self) -> float:
        """Return the cumulative irradiation dose of this turn (mJ)."""
        return float(self.turn.get("dose_mj") or 0.0)

    @property
    def errored(self) -> bool:
        """Return whether the turn ended with an error."""
        return bool(self.turn.get("error"))


    def state(self, path: str, when: str = "after"):
        """Read a before/after state value by dotted key."""
        return _dig(self.after if when == "after" else self.before, path)

    def unchanged(self, *paths: str, tol: float = 1e-9) -> bool:
        """Return whether a state value is unchanged (within tol)."""
        for p in paths:
            a, b = self.state(p, "before"), self.state(p, "after")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if abs(float(a) - float(b)) > tol:
                    return False
            elif a != b:
                return False
        return True

    def moved(self, axis: str, delta: float, tol: float = 0.001) -> bool:
        """Return whether a coordinate moved by exactly delta."""
        a, b = self.state(f"pos.{axis}", "before"), self.state(f"pos.{axis}", "after")
        if a is None or b is None:
            return False
        return abs((float(b) - float(a)) - delta) <= tol


    @property
    def results(self) -> list[dict]:
        """Return this session's measurements, including injected ones."""
        from backend.service.store.spectrum_store import list_results
        label = re.sub(r"[^0-9A-Za-z_-]", "-", self.sid)[:64]
        return [r for r in list_results(scope="session") if r["session"] == label]

    @property
    def new_results(self) -> list[dict]:
        """Return only the measurements made by the agent in this turn."""
        injected = {Path(n).stem for n in self.item["inputs"]}
        return [r for r in self.results
                if not any(stem in r["base"] for stem in injected)]

    @property
    def artifacts(self) -> list[dict]:
        """Return the artifacts in the run_store manifest."""
        from backend.service.store import run_store
        try:
            return run_store.list_artifacts()
        except Exception:
            return []

    def spectrum(self, base: str):
        """Read a saved measurement CSV as (x, y)."""
        from benchmark.test_pipeline.inject import read_spectrum_csv
        hit = next((r for r in self.results if r["base"] == base), None)
        if hit is None:
            raise KeyError(f"{base}: not in this session's measurement list")
        x, y, _ = read_spectrum_csv(Path(hit["csv"]))
        return x, y

    @property
    def new_files(self) -> set:
        return self.files_after - self.files_before

    def files_outside_session(self) -> list[Path]:
        """Return new files created outside the session-owned locations."""
        from backend.service.store import DATA_ROOT
        label = re.sub(r"[^0-9A-Za-z_-]", "-", self.sid)[:64]
        out = []
        for p in self.new_files:
            try:
                parts = p.relative_to(DATA_ROOT).parts
            except ValueError:
                out.append(p)
                continue
            root = parts[0] if parts else ""
            if root == "uploads":
                continue
            if root == "runs" and len(parts) > 1 and parts[1] == label:
                continue
            if root == "results" and len(parts) >= 4 and parts[2] == label:
                continue
            if root == "results" and len(parts) == 3 and parts[2].startswith("_"):
                continue
            out.append(p)
        return out


    @property
    def numbers(self) -> list[float]:
        """Return all numbers appearing in the answer."""
        text = _ISO_DATE_RE.sub(" ", _INV_CM_RE.sub(" cm ", self.answer or ""))
        out = []
        for m in _NUM_RE.finditer(text):
            try:
                out.append(float(m.group(0).replace(",", "")))
            except ValueError:
                pass
        return out

    def cites(self, value, tol: float = 1e-6, rel: bool = False) -> bool:
        """Return whether the answer cites a value within tolerance."""
        if value is None:
            return False
        v = float(value)
        limit = abs(v) * tol if rel else tol
        return any(abs(n - v) <= limit for n in self.numbers)

    def cites_any(self, values, tol: float = 1e-6, rel: bool = False) -> bool:
        """Return whether the answer cites any of the candidate values."""
        return any(self.cites(v, tol, rel) for v in values if v is not None)

    def cites_all(self, values, tol: float = 1e-6, rel: bool = False) -> bool:
        return all(self.cites(v, tol, rel) for v in values)

    def mentions(self, *tokens: str) -> bool:
        """Return whether any of the tokens appears in the answer."""
        text = re.sub(r"\s+", " ", (self.answer or "")).lower()
        return any(t.lower() in text for t in tokens)

    def claims_negative(self, term: str, window: int = 0) -> bool:
        """Return whether the answer describes term negatively (e.g. not connected)."""
        text = re.sub(r"[^\S\n]+", " ", (self.answer or "")).lower()
        t = term.lower()
        if window > 0:
            segs = [text[max(0, m.start() - window): m.end() + window]
                    for m in re.finditer(re.escape(t), text)]
        else:
            segs = [g for g in _SEGMENT_RE.split(text) if t in g]
        return any(w in g for g in segs for w in _NEGATIVE)

    def no_numbers_in(self, lo: float = PEAK_RANGE[0], hi: float = PEAK_RANGE[1],
                      exclude: tuple = ()) -> bool:
        """Return whether the answer contains no numbers in the given range."""
        return not any(lo <= n <= hi and all(abs(n - e) > 1e-6 for e in exclude)
                       for n in self.numbers)


if __name__ == "__main__":
    def _mk(answer="", calls=(), dose=0.0, before=None, after=None, item=None):
        return Run(item=item or {"id": "T00", "inputs": []}, sid="t",
                   before=before or {}, after=after or {}, answer=answer,
                   turn={"Tool calls": list(calls), "dose_mj": dose,
                         "tool_call_order": [c["name"] for c in calls]})

    C = lambda n, a=None, ok=True, res=None: {"name": n, "args": a or {}, "ok": ok,
                                              "result": res or {}}

    r = _mk(calls=[C("preview_grid_scan", {"rows": 3}), C("acquire_spectrum"),
                   C("run_grid_scan", {"rows": 3})])
    assert r.called("run_grid_scan") and not r.called("laser_on")
    assert r.not_called("laser_on", "laser_off")
    assert not r.not_called("acquire_spectrum")
    assert r.sequence("preview_grid_scan", "run_grid_scan")
    assert not r.sequence("run_grid_scan", "preview_grid_scan")
    assert r.args("preview_grid_scan")["rows"] == 3
    assert r.args("nope") == {}, "a missing tool must give an empty dict so .get is safe"

    r2 = _mk(calls=[C("acquire_spectrum", ok=False, res={"error": "no power"})])
    assert r2.failed("acquire_spectrum") and not r2.failed("move_stage")
    assert not _mk().failed("acquire_spectrum")
    r3 = _mk(calls=[C("do_3d_imaging", ok=False, res={"error": "Unknown tool: do_3d_imaging"})])
    assert r3.unknown_tools() == ["do_3d_imaging"]

    b = {"pos": {"x": 37.0, "y": 25.0, "z": 0.1}, "ccd": {"shutter_mode": "auto"}}
    a = {"pos": {"x": 37.05, "y": 25.0, "z": 0.1}, "ccd": {"shutter_mode": "close"}}
    r = _mk(before=b, after=a)
    assert r.state("ccd.shutter_mode") == "close"
    assert r.state("ccd.shutter_mode", "before") == "auto"
    assert r.state("ccd.no_key") is None and r.state("no.path") is None
    assert r.unchanged("pos.y", "pos.z") and not r.unchanged("pos.x")
    assert r.moved("x", +0.05) and not r.moved("x", -0.05)
    assert r.moved("y", 0.0)

    r = _mk("The maximum intensity is 4,411.0 ADU and the SNR is 142.88. Power 5%.")
    assert r.cites(4411.0) and r.cites(5) and r.cites(142.88, tol=0.01)
    assert not r.cites(9999)
    assert r.cites(143.06, rel=True, tol=0.01), "within 1% relative error must pass"
    assert r.cites_any([142.884525, 143.063019], tol=0.01)
    assert not r.cites_any([1.0, 2.0], tol=0.01)
    assert r.cites_all([4411.0, 5])

    r = _mk(r"Peaks at 1001 cm-1 and $1350 \text{ cm}^{-1}$, saved to results/2026-08-24/x.csv")
    assert r.cites(1001) and r.cites(1350), "real peaks must remain"
    assert not r.cites(-1), "the -1 in cm-1 is not a number"
    assert not r.cites(2026) and not r.cites(-8) and not r.cites(-24), "dates in paths are not numbers"

    r = _mk("Z range is limited to -1.0 mm to 1.0 mm; dose is **5.7** mJ; SNR 7.07 x 10^5")
    assert r.cites(-1.0), "reading 'to -1.0' as a unit would remove the correct answer of X04"
    assert r.cites(5.7), "reading markdown bold (**) as a power would remove the correct answer of D16"
    assert r.cites(7.07) and r.cites(10), "removing exponent notation would break scientific notation"

    r = _mk("This looks like PHOTOBLEACHING of the sample.")
    assert r.mentions("photobleach") and not r.mentions("saturation")

    for _tag, _ans in (
        ("one line", "stage: connected. laser: connected. ccd: NOT CONNECTED. camera: connected."),
        ("line breaks", "stage - connected\nlaser - connected\nccd - not connected\ncamera - connected"),
        ("table", "| Device | Status |\n|---|---|\n| stage | connected |\n| laser | connected |\n"
               "| ccd | unavailable |\n| camera | connected |"),
        ("bullets", "- stage: OK\n- laser: OK\n- ccd: not detected\n- camera: OK"),
        ("prose", "The stage is connected, the laser is connected, the ccd is not connected, the camera is connected"),
    ):
        _r = _mk(_ans)
        _neg = [d for d in ("stage", "laser", "ccd", "camera") if _r.claims_negative(d)]
        assert _neg == ["ccd"], f"{_tag}: negative report is {_neg}"
    assert not _mk("All four are connected: stage, laser, ccd, camera.").claims_negative("ccd")

    assert _mk("The file does not exist. 0 files.").no_numbers_in()
    assert not _mk("The highest peak is at 1001 cm-1.").no_numbers_in()
    assert _mk("All values are the same constant 1234.").no_numbers_in(exclude=(1234.0,))

    assert _mk(dose=0.0).dose == 0.0 and _mk(dose=0.19).dose == 0.19

    print("passed: 8 behaviours · 6 states · 6 number citations (incl. ddof) · tokens · fabricated numbers · dose · spurious numbers blocked")
