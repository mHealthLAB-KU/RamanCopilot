"""Checks that declared band sources appeared in the current session."""

from __future__ import annotations

import re
import threading
from typing import Iterable

from backend.repo.store import run_store

_LOCK = threading.Lock()
_STATE: dict[str, dict] = {}
_MAX_SESSIONS = 32  # sessions kept in memory
USER_TOL_CM1 = 2.0  # tolerance between a band and a number in the user message


def _key() -> str:
    return run_store.current().get("label") or "_unassigned"


def _st() -> dict:
    with _LOCK:
        k = _key()
        s = _STATE.get(k)
        if s is None:
            if len(_STATE) >= _MAX_SESSIONS:
                _STATE.pop(next(iter(_STATE)), None)
            s = _STATE[k] = {"user": [], "urls": {}}
        return s


def _norm(url: str) -> str:
    u = (url or "").strip().strip("<>()[]'\",;")
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.rstrip("/").lower()


def note_user_message(text: str) -> None:
    """Record the user message of this turn."""
    if text:
        _st()["user"].append(str(text))


def note_web_result(result: dict) -> None:
    """Collect URLs from a web_search result."""
    if not isinstance(result, dict) or not result.get("ok", True):
        return
    urls = _st()["urls"]
    for r in result.get("results") or []:
        if isinstance(r, dict) and r.get("url"):
            urls[_norm(r["url"])] = str(r.get("title") or "")[:200]


def known_urls() -> list[str]:
    """Return the URLs returned by web search in the current session."""
    return list(_st()["urls"])


def _numbers(texts: Iterable[str]) -> list[float]:
    out = []
    for t in texts:
        out += [float(m) for m in re.findall(r"(?<![\d.])(\d{2,4}(?:\.\d+)?)(?![\d.])", t)]
    return out


def check(source: str, centers: list[float]) -> tuple[bool, str]:
    """Return (passed, reason); on success the reason is the normalized source."""
    src = (source or "").strip()
    low = src.lower()
    if low == "user":
        nums = _numbers(_st()["user"])
        missing = [c for c in centers if not any(abs(c - n) <= USER_TOL_CM1 for n in nums)]
        if missing:
            return False, (f"bands_source='user' but these band positions do not appear in anything the user wrote "
                           f"in this session: {missing}. Do not label your own knowledge as the user's. Find the "
                           f"bands in the literature with web_search and give bands_source='web:<url>' (a URL that "
                           f"search returned), or ask the user.")
        return True, "user"
    if low.startswith("web:"):
        given = [re.sub(r"^web:", "", u, flags=re.I) for u in re.split(r"[\s,]+", src[4:]) if u.strip()]
        given = [u for u in given if u]
        if not given:
            return False, "bands_source='web:' needs at least one URL, e.g. 'web:https://...'."
        have = _st()["urls"]
        bad = [u for u in given if not any(_norm(u) == h or (len(_norm(u)) > 12 and (h.startswith(_norm(u))
                                                                                   or _norm(u).startswith(h)))
                                           for h in have)]
        if bad:
            return False, (f"These URLs were not returned by any web_search in this session: {bad}. Cite only "
                           f"pages your search actually returned (known: {list(have)[:8]}). Run web_search first "
                           f"if you have not.")
        return True, "web:" + " ".join(given)
    return False, (f"bands_source='{src}' is not an accepted source. The text knowledge base was removed on "
                   f"2026-09-06, and learned skills / memory are not literature. Accepted: 'user' (the user gave "
                   f"these numbers in this session) or 'web:<url>' (a page returned by web_search in this session). "
                   f"Search the literature for this sample's Raman bands (positions AND widths) first.")
