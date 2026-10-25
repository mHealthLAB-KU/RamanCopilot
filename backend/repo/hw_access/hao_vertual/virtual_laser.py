"""Virtual laser that tracks beam state and power."""

from __future__ import annotations


class _DummySerial:
    """Stub serial port that passes the `laser.ser.is_open` connection check."""

    is_open = True

    def close(self) -> None:
        self.is_open = False


class VirtualLaser:
    """Virtual laser holding only beam state and power."""

    ND_MIN_PCT, ND_MAX_PCT = 0.004, 100.0

    def __init__(self, port: str = "VIRTUAL", baud: int = 115200):
        self.port = port
        self.ser = _DummySerial()
        self._power_set = False
        self.power_pct = None
        self.is_on = False
        self.bs_signal = False

    def laser_on(self) -> bool:
        if self._power_set:
            self.bs_signal = True
        self.is_on = True
        return True

    def laser_off(self) -> bool:
        """Turn the laser off; always confirmed in the virtual laser."""
        self.is_on = False
        return True

    def set_guide_beam(self) -> bool:
        """Move the ND filter to the blocking position (disarms the measurement beam)."""
        self._power_set = False
        self.bs_signal = False
        return True

    def set_power(self, percent) -> bool:
        """Set the power, clipping to range and returning False on failure like the real controller."""
        try:
            p = float(percent)
        except (TypeError, ValueError):
            return False
        self.power_pct = max(self.ND_MIN_PCT, min(self.ND_MAX_PCT, p))
        self._power_set = True
        return True

    def pulse_to_percent(self, pulse):
        """Return the value unchanged (the virtual laser has no ND motor)."""
        return float(pulse)

    def home_search(self) -> bool:
        return True

    def close(self) -> None:
        self.laser_off()
        self.set_guide_beam()
        self.ser.close()

    @property
    def beam(self) -> str:
        """Return the beam state: 'off', 'guide' or 'measurement'."""
        if not self.is_on:
            return "off"
        return "measurement" if self._power_set else "guide"

    def __repr__(self) -> str:
        return f"<VirtualLaser {self.beam} power={self.power_pct}%>"
