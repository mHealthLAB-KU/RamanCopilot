"""Camera frame preprocessing and guide-beam spot measurement."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.common.config.hw import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_SENSOR_BIN
from backend.common import optics_map


def to_uint8_gray(frame) -> "np.ndarray":
    """Convert a frame to 8-bit grayscale."""
    img = np.asarray(frame)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _view_window(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Return the view window (x0, y0, w, h) within the sensor frame, in sensor pixels."""
    cw, ch = CAMERA_WIDTH * CAMERA_SENSOR_BIN, CAMERA_HEIGHT * CAMERA_SENSOR_BIN
    if frame_w >= cw and frame_h >= ch:
        return ((frame_w - cw) // 2, (frame_h - ch) // 2, cw, ch)
    return (0, 0, frame_w, frame_h)


def sensor_to_view_px(x: float, y: float, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Convert sensor-frame coordinates to view-resolution (tool) coordinates."""
    x0, y0, cw, ch = _view_window(frame_w, frame_h)
    return ((x - x0) * CAMERA_WIDTH / cw, (y - y0) * CAMERA_HEIGHT / ch)


def to_view_bgr(frame, width: int | None = None, height: int | None = None) -> "np.ndarray":
    """Convert a frame to 8-bit BGR at the view resolution used as the tool coordinate system."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    img = np.asarray(frame)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    x0, y0, cw, ch = _view_window(img.shape[1], img.shape[0])
    img = img[y0:y0 + ch, x0:x0 + cw]
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if bgr.shape[:2] != (h, w):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


AIM_RING_R_PX = 14  # radius of the laser spot ring

def _aim_center(img, center) -> tuple[int, int]:
    """Return the laser spot pixel used for the overlay ring."""
    h, w = img.shape[:2]
    x, y = center if center is not None else optics_map.aim_px(w, h)
    return (int(round(x)), int(round(y)))


AIM_GAP_PX = 22  # gap between the ring and the crosshair


def annotate_aim(bgr, center=None) -> "np.ndarray":
    """Return a new image with the laser spot crosshair and ring drawn on it."""
    out = np.asarray(bgr).copy()
    h, w = out.shape[:2]
    cx, cy = _aim_center(out, center)
    red = (0, 0, 255)

    segments = [((cx, 0), (cx, cy - AIM_GAP_PX)),
                ((cx, cy + AIM_GAP_PX), (cx, h)),
                ((0, cy), (cx - AIM_GAP_PX, cy)),
                ((cx + AIM_GAP_PX, cy), (w, cy))]
    for p0, p1 in segments:
        cv2.line(out, p0, p1, red, 1, cv2.LINE_AA)

    cv2.circle(out, (cx, cy), AIM_RING_R_PX, red, 2, cv2.LINE_AA)
    return out


MODEL_IMAGE_MAX_SIDE = 1568  # px, longest side of images sent to the model
MODEL_JPEG_QUALITY = 85  # JPEG quality of those images


def encode_for_model(bgr, max_side: int = MODEL_IMAGE_MAX_SIDE, quality: int = MODEL_JPEG_QUALITY) -> str | None:
    """Encode an image as a base64 JPEG for the model."""
    img = np.asarray(bgr)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ret:
        return None
    return base64.b64encode(buf).decode('utf-8')


def sharpness_score(gray) -> float:
    """Return the variance of the Laplacian as a relative sharpness score."""
    return float(np.var(cv2.Laplacian(np.asarray(gray).astype(np.float32), cv2.CV_32F)))


def _flush_frames(camera, n: int = 3) -> None:
    """Discard frames buffered before a laser state change."""
    for _ in range(n):
        camera.get_latest_frame()


def _mean_frame(camera, n: int = 3):
    """Return the mean of n consecutive 8-bit gray frames, or None if none were received."""
    frames = []
    for _ in range(n):
        f = camera.get_latest_frame()
        if f is not None:
            frames.append(to_uint8_gray(f))
    if not frames:
        return None
    return np.mean(frames, axis=0).astype(np.uint8)


def _spot_center(blurred, binary):
    """Return the center of the main spot in a spot mask (sensor pixels), or None."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n < 2:
        return None
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    weight = blurred.astype(np.float64) * (labels == main)
    total = weight.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(weight.shape)
    return (float((weight * xs).sum() / total), float((weight * ys).sum() / total))


def capture_laser_diff(camera, laser, n_avg: int = 3) -> dict:
    """Isolate the beam spot by laser OFF/ON frame difference and measure its area and center."""
    laser.laser_off()
    _flush_frames(camera, n_avg)
    ref = _mean_frame(camera, n_avg)

    laser.laser_on()
    _flush_frames(camera, n_avg)
    lit = _mean_frame(camera, n_avg)

    if ref is None or lit is None:
        return {"ok": False, "area_px": 0, "spot_px": None, "frame_size": None,
                "ref": ref, "laser_frame": lit, "diff_clip": None, "diff_absdiff": None}

    diff_clip = np.clip(lit.astype(np.int16) - ref.astype(np.int16), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(diff_clip, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"ok": True, "area_px": int(np.count_nonzero(binary)),
            "spot_px": _spot_center(blurred, binary),
            "frame_size": (diff_clip.shape[1], diff_clip.shape[0]),
            "ref": ref, "laser_frame": lit,
            "diff_clip": diff_clip, "diff_absdiff": cv2.absdiff(lit, ref)}


def guide_beam_spot_area(camera, laser, n_avg: int = 3):
    """Return the guide-beam spot area in pixels (autofocus objective), or None if no frame was received."""
    d = capture_laser_diff(camera, laser, n_avg)
    if not d.get("ok"):
        return None
    return int(d.get("area_px", 0))
