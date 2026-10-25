"""Virtual microscope camera that renders the virtual sample at the stage position."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from backend.common.config.hw import CAMERA_HEIGHT, CAMERA_WIDTH

_REF_EXPOSURE_MS = 50.0  # exposure at which the image has its original brightness

_AUTO_TARGET_MEAN = 118.0  # mean brightness targeted by auto exposure, 0-255

_SPOT_RADIUS_PX = 26  # guide-beam spot radius
_SPOT_GAIN = 90.0  # guide-beam spot brightness


class VirtualCamera:
    """Virtual camera that renders frames by cropping the virtual sample."""

    def __init__(self, mgr, scene, exposure_ms: float = _REF_EXPOSURE_MS):
        """Create a virtual camera from the hardware manager and the virtual sample."""
        self._mgr = mgr
        self.scene = scene
        self.exposure_ms = float(exposure_ms)
        self.auto_exposure = False
        self.is_streaming = False
        self._lock = threading.RLock()

    def start_stream(self) -> None:
        with self._lock:
            self.is_streaming = True

    def stop_stream(self) -> None:
        with self._lock:
            self.is_streaming = False

    def close(self) -> None:
        self.stop_stream()

    def set_exposure(self, ms: float) -> None:
        """Set the exposure; this disables auto exposure, as on the real camera."""
        with self._lock:
            self.exposure_ms = max(0.01, float(ms))
            self.auto_exposure = False

    def set_auto_exposure(self, enabled: bool) -> None:
        with self._lock:
            self.auto_exposure = bool(enabled)

    def get_latest_frame(self):
        """Return the latest (H, W, 3) uint8 BGR frame, or None if not streaming."""
        with self._lock:
            if not self.is_streaming:
                return None
            view = self.scene.crop(*self._stage_xy(), CAMERA_WIDTH, CAMERA_HEIGHT)
            view = self._apply_brightness(view)
            view = self._apply_beam(view)
            return self._apply_noise(view)

    def _stage_xy(self) -> tuple[float, float]:
        """Return the current stage position, or the sample center if no stage is available."""
        stage = getattr(self._mgr, "stage", None)
        pos = stage.get_position() if stage is not None else None
        if pos is None:
            return self.scene.center_mm
        return (pos[0], pos[1])

    def _apply_brightness(self, view):
        gain = self.exposure_ms / _REF_EXPOSURE_MS
        if self.auto_exposure:
            mean = float(view.mean())
            if mean > 1.0:
                gain = _AUTO_TARGET_MEAN / mean
        if abs(gain - 1.0) < 1e-6:
            return view
        return np.clip(view.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    def _apply_beam(self, view):
        """Draw the laser spot at the image center when the laser is on."""
        laser = getattr(self._mgr, "laser", None)
        if laser is None or not getattr(laser, "is_on", False):
            return view
        h, w = view.shape[:2]
        spot = np.zeros((h, w), np.float32)
        cv2.circle(spot, (w // 2, h // 2), _SPOT_RADIUS_PX, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), _SPOT_RADIUS_PX / 3.0)
        armed = bool(getattr(laser, "_power_set", False))
        strength = _SPOT_GAIN * (1.0 + (float(getattr(laser, "power_pct", 0) or 0) / 100.0)) if armed else _SPOT_GAIN * 0.5
        out = view.astype(np.float32) + spot[:, :, None] * strength
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_noise(self, view):
        noise = np.random.normal(0.0, 2.0, view.shape).astype(np.float32)
        return np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:
        mode = "auto" if self.auto_exposure else f"{self.exposure_ms:.1f}ms"
        return f"<VirtualCamera {'streaming' if self.is_streaming else 'idle'} exp={mode}>"
