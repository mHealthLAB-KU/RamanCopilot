
from __future__ import annotations

import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
ANS_DIR = _ROOT / "benchmark" / "ans"


DEV_REL = 1e-3
DEV_ABS = 1e-6


SPEC_DEVIATIONS = {
    "S03": (
        "Spec says `exposure_time_s == 2.0 (1e-6)`. The CCD cannot store 2.0: the Andor "
        "SDK quantises exposure to the readout clock and the driver reads the applied "
        "value back (measured on this instrument: 1.0 -> 0.99999, 0.5 -> 0.50001). "
        "Enforcing 1e-6 would make the item unpassable by any agent, so it is graded "
        "with the device tolerance (relative 1e-3) instead. Every other item whose spec "
        "states a tolerance is graded exactly as written."
    ),
}


def near(got, want, rel: float = DEV_REL, abs_: float = DEV_ABS) -> bool:
    """Return whether a value read back from the instrument equals the commanded value."""
    if got is None or want is None:
        return False
    try:
        return math.isclose(float(got), float(want), rel_tol=rel, abs_tol=abs_)
    except (TypeError, ValueError):
        return False


def saved_paths(r, kind: str = "spectra") -> list[Path]:
    """Return absolute paths of artifacts saved by the agent in this session, in save order."""
    from backend.service.store import DATA_ROOT
    return [DATA_ROOT / a["path"] for a in r.artifacts if a.get("kind") == kind]


def agent_spectra(r) -> list[tuple[list, list]]:
    """Return all spectra saved by the agent as (x, y)."""
    from benchmark.test_pipeline.inject import read_spectrum_csv
    out = []
    for p in saved_paths(r, "spectra"):
        if p.suffix.lower() != ".csv" or not p.is_file():
            continue
        x, y, _ = read_spectrum_csv(p)
        out.append((x, y))
    return out


def expected(item: dict) -> tuple[list, list]:
    """Return the reference arrays from ans/<expected_file>, or ([], [])."""
    from benchmark.test_pipeline.inject import read_spectrum_csv
    name = item.get("expected_file")
    if not name:
        return [], []
    x, y, _ = read_spectrum_csv(ANS_DIR / name)
    return x, y


def input_spectrum(item: dict, name: str | None = None) -> tuple[list, list]:
    """Return an item's input CSV as (x, y)."""
    from benchmark.test_pipeline.inject import INPUT_DIR, read_spectrum_csv
    target = name or (item["inputs"][0] if item["inputs"] else None)
    if not target:
        return [], []
    x, y, _ = read_spectrum_csv(INPUT_DIR / target)
    return x, y


def cos(a, b) -> float:
    """Cosine similarity (0 if lengths differ)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(p * q for p, q in zip(a, b)) / (na * nb)


def nrmse(a, b) -> float:
    """RMSE normalized by the range of the reference (inf if lengths differ)."""
    if not a or not b or len(a) != len(b):
        return float("inf")
    rng = max(b) - min(b)
    if rng == 0:
        return float("inf")
    mse = sum((p - q) ** 2 for p, q in zip(a, b)) / len(b)
    return math.sqrt(mse) / rng


def allclose(a, b, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """Return whether lengths and all values match."""
    if len(a) != len(b):
        return False
    return all(abs(p - q) <= atol + rtol * abs(q) for p, q in zip(a, b))


def matches_expected(r, item, rtol: float = 1e-6) -> tuple[bool, str]:
    """Return whether any spectrum saved by the agent matches the reference."""
    ex_x, ex_y = expected(item)
    if not ex_y:
        return False, f"no reference file ({item.get('expected_file')})"
    got = agent_spectra(r)
    if not got:
        return False, "no spectrum was saved"
    for _x, y in got:
        if allclose(y, ex_y, rtol):
            return True, ""
    best = max((cos(y, ex_y) for _x, y in got), default=0.0)
    return False, f"does not match the reference ({len(got)} saved, best cos {best:.5f})"


def close_enough(r, item, min_cos: float = 0.99, max_nrmse: float = 0.02) -> tuple[bool, str]:
    """Shape-based match for items that allow several methods."""
    ex_x, ex_y = expected(item)
    if not ex_y:
        return False, f"no reference file ({item.get('expected_file')})"
    for _x, y in agent_spectra(r):
        if cos(y, ex_y) >= min_cos and nrmse(y, ex_y) <= max_nrmse:
            return True, ""
    return False, f"shape differs from the reference (needs cos >= {min_cos} and NRMSE <= {max_nrmse})"


def in_range(x, y, lo: float, hi: float) -> tuple[list, list]:
    """Return the part within [lo, hi] (inclusive)."""
    pairs = [(a, b) for a, b in zip(x, y) if lo <= a <= hi]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def stdev_both(vals) -> list[float]:
    """Return both sample (ddof=1) and population (ddof=0) standard deviations."""
    n = len(vals)
    if n == 0:
        return []
    m = sum(vals) / n
    ss = sum((v - m) ** 2 for v in vals)
    out = [math.sqrt(ss / n)]
    if n > 1:
        out.append(math.sqrt(ss / (n - 1)))
    return out


def snr_both(x, y, lo: float = 1800.0, hi: float = 2000.0) -> list[float]:
    """SNR = maximum / noise-region standard deviation, for both ddof values."""
    _nx, ny = in_range(x, y, lo, hi)
    if not y or not ny:
        return []
    peak = max(y)
    return [peak / s for s in stdev_both(ny) if s > 0]


def trapz(x, y) -> float:
    """Trapezoidal integral."""
    return sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0 for i in range(len(x) - 1))


def argmax_in(x, y, lo: float, hi: float):
    """Return x of the maximum intensity within [lo, hi], or None."""
    xs, ys = in_range(x, y, lo, hi)
    if not ys:
        return None
    return xs[max(range(len(ys)), key=lambda i: ys[i])]


def max_in(x, y, lo: float, hi: float):
    """Return the maximum intensity within [lo, hi], or None."""
    _xs, ys = in_range(x, y, lo, hi)
    return max(ys) if ys else None


def pearson(a, b) -> float:
    """Pearson correlation coefficient."""
    n = len(a)
    if n < 2 or len(b) != n:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    da = math.sqrt(sum((p - ma) ** 2 for p in a))
    db = math.sqrt(sum((q - mb) ** 2 for q in b))
    return num / (da * db) if da and db else 0.0


def set_close(got, want, tol: float) -> bool:
    """Return whether two value sets match one-to-one within tol."""
    if len(got) != len(want):
        return False
    left = list(want)
    for g in got:
        hit = next((w for w in left if abs(g - w) <= tol), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


if __name__ == "__main__":
    assert abs(cos([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
    assert abs(cos([1, 0], [0, 1])) < 1e-12
    assert cos([1, 2], [1, 2, 3]) == 0.0, "different lengths cannot be compared = failure"
    assert nrmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert nrmse([1, 2], [1, 2, 3]) == float("inf")

    assert allclose([1.0, 2.0], [1.0, 2.0])
    assert not allclose([1.0, 2.0], [1.0, 2.1])
    assert not allclose([1.0], [1.0, 2.0])

    x = [float(i) for i in range(0, 11)]
    y = [1.0] * 11
    assert abs(trapz(x, y) - 10.0) < 1e-12, "width 10, height 1 = area 10"
    assert in_range(x, y, 2, 4)[0] == [2.0, 3.0, 4.0], "both ends included"
    assert argmax_in(x, [0, 5, 1, 9, 2, 0, 0, 0, 0, 0, 0], 0, 4) == 3.0
    assert max_in(x, [0, 5, 1, 9, 2, 0, 0, 0, 0, 0, 0], 0, 4) == 9

    s = stdev_both([1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(s) == 2 and s[1] > s[0]
    assert abs(s[0] - math.sqrt(2.0)) < 1e-12 and abs(s[1] - math.sqrt(2.5)) < 1e-12
    assert abs(s[1] / s[0] - 1.118) < 0.001, "about 12% difference at n=5"
    assert stdev_both([]) == [] and len(stdev_both([1.0])) == 1

    assert abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-12
    assert abs(pearson([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-12

    assert set_close([1001.2, 620.0], [620.0, 1001.0], tol=3.0)
    assert not set_close([1001.2], [620.0, 1001.0], tol=3.0), "different counts fail"
    assert not set_close([500.0, 620.0], [620.0, 1001.0], tol=3.0)

    assert near(0.99999, 1.0) and near(0.50001, 0.5) and near(1.99998, 2.0)
    assert not near(0.5, 1.0) and not near(1.0, 2.0), "neighbouring exposure values must still differ"
    assert not near(None, 1.0) and not near(1.0, None) and not near("x", 1.0)
    assert near(0.0, 0.0) and not near(0.002, 0.0), "near zero, an absolute tolerance floor applies"

    print("passed: 3 similarity measures · integral/range/argmax · both ddof · correlation · set matching · "
          "instrument quantization absorbed")
