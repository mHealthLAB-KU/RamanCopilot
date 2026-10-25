
from __future__ import annotations

import importlib

_AXES = "SCPAXDL"

GRADERS: dict = {}
for _k in _AXES:
    _mod = importlib.import_module(f"{__name__}.{_k}")
    for _name in dir(_mod):
        if len(_name) == 3 and _name[0] == _k and _name[1:].isdigit():
            GRADERS[_name] = getattr(_mod, _name)


def checkpoints(item: dict, run) -> list[tuple[bool, str]]:
    """Grade one item and return [(passed, description), ...]."""
    fn = GRADERS.get(item["id"])
    if fn is None:
        return []
    try:
        out = []
        for cp in fn(run):
            out.append(cp if isinstance(cp, tuple) else (bool(cp), ""))
        return out
    except Exception as e:  # noqa: BLE001
        return [(False, f"grader raised: {type(e).__name__}: {e}")]


def selfcheck() -> str:
    """Check that the set of graders matches items.json exactly."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    items = json.loads((root / "benchmark" / "items.json").read_text(encoding="utf-8"))["items"]
    ids = {i["id"] for i in items}
    human = {i["id"] for i in items if i["needs_human"]}

    orphan = sorted(set(GRADERS) - ids)
    assert not orphan, f"graders for items not in items.json: {orphan}"
    missing = sorted(ids - set(GRADERS) - human)
    assert not missing, f"items without a grader: {missing}"
    both = sorted(set(GRADERS) & human)
    assert not both, f"human-graded items that also have a grader: {both}"

    import re as _re
    from pathlib import Path as _P
    _han = _re.compile(r"[\uac00-\ud7a3]")
    leaks = []
    for _f in sorted(_P(__file__).parent.glob("*.py")):
        for _n, _line in enumerate(_f.read_text(encoding="utf-8").splitlines(), 1):
            if "yield" not in _line or _line.strip().startswith("#"):
                continue
            if "mentions(" in _line or "claims_negative" in _line:
                continue
            for _m in _re.finditer(r'f?"[^"]*"', _line):
                if _han.search(_m.group(0)):
                    leaks.append(f"{_f.name}:{_n} {_m.group(0)[:60]}")
    assert not leaks, ("Korean text left in checkpoint descriptions (result data must be in English):\n  "
                       + "\n  ".join(leaks))

    from benchmark.test_pipeline.graders import _util as _U
    stray = sorted(set(_U.SPEC_DEVIATIONS) - ids)
    assert not stray, f"spec deviations for items not in items.json: {stray}"
    import re as _re2
    assert not any(_re2.search(r"[\uac00-\ud7a3]", v) for v in _U.SPEC_DEVIATIONS.values()),         "spec deviation notes appear in summary.md - they must be in English"

    return (f"{len(GRADERS)} graders · {len(human)} human-graded "
            f"({', '.join(sorted(human))}) · all descriptions in English · "
            f"total {len(GRADERS) + len(human)} == {len(ids)} · "
            f"{len(_U.SPEC_DEVIATIONS)} spec deviations ({', '.join(sorted(_U.SPEC_DEVIATIONS))})")
