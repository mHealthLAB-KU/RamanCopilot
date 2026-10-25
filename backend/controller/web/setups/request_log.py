"""Console logging of requests and instrument status changes."""
from __future__ import annotations

import time


POLLING_PATHS = {"/api/hw/status"}

_last_polling_status: dict[str, int] = {}


def should_log_request(path: str, status: int) -> bool:
    """Decide whether to log this response; polling paths are logged only when the status code changes."""
    if path not in POLLING_PATHS:
        return True
    if _last_polling_status.get(path) == status:
        return False
    _last_polling_status[path] = status
    return True


def _router_label(scope) -> str:
    """Return the name of the router that handled the request, or an empty string."""
    tags = getattr(scope.get("route"), "tags", None)
    return f"  [{tags[0]}]" if tags else ""


class RequestLogMiddleware:
    """Middleware that logs one line per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        started = time.perf_counter()
        path = scope.get("path", "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                if should_log_request(path, status):
                    ms = (time.perf_counter() - started) * 1000
                    print(f"[REQ] {scope.get('method', '?'):6} {path} "
                          f"-> {status}  ({ms:.0f}ms){_router_label(scope)}")
            await send(message)

        await self.app(scope, receive, send_wrapper)


CCD_LOG_STEP_C = 1.0  # smallest CCD temperature change that is logged

_HW_LOG_SKIP = {"stage.x", "stage.y", "stage.z", "ccd.temperature", "ccd.preamp_gains",
                "ccd.vs_speeds_us", "ccd.hs_speeds_mhz", "laser.power_range", "laser.power_pct",
                "laser.full_mw", "optics.available",
                "optics.aim.width", "optics.aim.height"}

_hw_last: dict = {}


def _flatten(d, prefix="") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def log_hw_status(status: dict) -> bool:
    """Log the hardware fields that changed since the last observation; return True if anything was logged."""
    now = _flatten(status)
    first = not _hw_last
    changes = [f"{k} {_hw_last.get(k)}→{v}" for k, v in now.items()
               if k not in _HW_LOG_SKIP and _hw_last.get(k) != v and not first]

    t_now, t_last = now.get("ccd.temperature"), _hw_last.get("ccd.temperature")
    if t_now is not None and (t_last is None or abs(t_now - t_last) >= CCD_LOG_STEP_C):
        changes.append(f"ccd.temperature {t_last}→{t_now} ({now.get('ccd.temp_status')})")
        _hw_last["ccd.temperature"] = t_now
    temp_kept = _hw_last.get("ccd.temperature")
    _hw_last.clear()
    _hw_last.update(now)
    _hw_last["ccd.temperature"] = temp_kept if t_now is not None else None

    if first:
        on = [k for k in ("stage", "ccd", "camera", "laser") if now.get(f"connected.{k}")]
        print(f"[HW] connected: {', '.join(on) or 'none'}")
        return True
    if not changes:
        return False
    print("[HW] " + " · ".join(changes))
    return True
