"""Virtual XY stage that tracks position only."""

from __future__ import annotations

from backend.common.config.hw import (
    STAGE_CENTER_X, STAGE_CENTER_Y, STAGE_MAX_X, STAGE_MAX_Y,
)

from backend.repo.hw_access.hao.USE_stage_test import MAX_SPEED_XY, MAX_SPEED_Z


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


class VirtualStage:
    """Virtual stage holding only position and velocity."""

    has_z = False

    def __init__(self, dll_path: str | None = None,
                 x: float | None = None, y: float | None = None):
        self.x = STAGE_CENTER_X if x is None else float(x)
        self.y = STAGE_CENTER_Y if y is None else float(y)
        self.z = 0.0
        self.a = 0.0
        self.vx = self.vy = self.vz = self.va = 0.1
        self.connected = False
        self.dead = False
        self.dead_reason = None
        self.LSID = -1

    def load_dll(self) -> bool:
        return True

    def create_session(self) -> bool:
        self.LSID = 1
        return True

    def connect(self, interface: int = -1, port: str = "", baudrate: int = 57600) -> bool:
        self.connected = True
        return True

    def calibrate(self, axes: int = 3) -> bool:
        return True

    def mark_dead(self, reason: str) -> None:
        """Mark the session as unrecoverable, as the real stage does on a failed DLL release."""
        self.dead = True
        self.dead_reason = reason

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def free_session(self) -> bool:
        self.LSID = -1
        return True

    def get_position(self):
        """Return (x, y, z, a), or None on failure like the real stage."""
        if self.dead or not self.connected:
            return None
        return (self.x, self.y, self.z, self.a)

    def move_absolute(self, x: float, y: float, z: float = 0.0, a: float = 0.0,
                      wait: bool = True) -> bool:
        """Move to an absolute position (z is ignored), clipped to the travel range."""
        if self.dead or not self.connected:
            return False
        cx, cy = _clip(x, 0.0, STAGE_MAX_X), _clip(y, 0.0, STAGE_MAX_Y)
        if (cx, cy) != (float(x), float(y)):
            print(f"[WARN] out of range -> clipping: "
                  f"X {float(x):.4f}->{cx:.4f}  Y {float(y):.4f}->{cy:.4f}")
        self.x, self.y = cx, cy
        return True

    def move_relative(self, dx: float, dy: float, dz: float = 0.0, da: float = 0.0,
                      wait: bool = True) -> bool:
        return self.move_absolute(self.x + dx, self.y + dy, 0.0, 0.0, wait)

    def get_velocity(self) -> dict:
        if self.dead:
            return {"ok": False, "error": f"session invalid - {self.dead_reason}"}
        if not self.connected:
            return {"ok": False, "error": "not connected"}
        return {"ok": True,
                "x_speed_mm_s": self.vx, "y_speed_mm_s": self.vy,
                "z_speed_mm_s": self.vz, "a_speed_mm_s": self.va}

    def set_velocity(self, vx: float, vy: float, vz: float, va: float) -> bool:
        if self.dead or not self.connected:
            return False
        self.vx = _clip(vx, 0.0, MAX_SPEED_XY)
        self.vy = _clip(vy, 0.0, MAX_SPEED_XY)
        self.vz = _clip(vz, 0.0, MAX_SPEED_Z)
        self.va = _clip(va, 0.0, MAX_SPEED_XY)
        return True

    def __repr__(self) -> str:
        return f"<VirtualStage x={self.x:.4f} y={self.y:.4f} (no Z)>"
