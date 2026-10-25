"""Request bodies for the web API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from backend.common.config.hw import CAMERA_EXPOSURE_MS, CAMERA_HEIGHT, CAMERA_WIDTH, LASER_PORT
from backend.service.tools.hw_tools.acquire_tools import acquire_spectrum
from backend.service.tools.schema import public_args_model


class CameraConnectRequest(BaseModel):
    """Body of POST /api/camera/connect."""
    exposure_ms: float = CAMERA_EXPOSURE_MS


class LaserConnectRequest(BaseModel):
    """Body of POST /api/laser/connect."""
    port: str = LASER_PORT


class MovePixelRequest(BaseModel):
    """Body of POST /api/stage/move-pixel."""
    px: float
    py: float
    stream_width: int = CAMERA_WIDTH
    stream_height: int = CAMERA_HEIGHT


class CcdSettings(BaseModel):
    """CCD fields of the settings panel."""
    exposure_s: Optional[float] = None
    acq_mode: Optional[str] = None
    num_acc: Optional[int] = None
    num_kin: Optional[int] = None
    kinetic_cycle_s: Optional[float] = None
    read_mode: Optional[str] = None
    hbin: Optional[int] = None
    track_center: Optional[int] = None
    track_width: Optional[int] = None
    trigger: Optional[str] = None
    shutter: Optional[str] = None
    preamp_index: Optional[int] = None
    vs_index: Optional[int] = None
    hs_index: Optional[int] = None
    target_temp: Optional[int] = None
    cooler: Optional[bool] = None


class LaserSettings(BaseModel):
    """Laser fields of the settings panel."""
    power_mw: Optional[float] = None


class CameraSettings(BaseModel):
    """Camera fields of the settings panel."""
    exposure_ms: Optional[float] = None
    auto_exposure: Optional[bool] = None


class StageSpeed(BaseModel):
    """Per-axis stage speed in mm/s."""
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class StageSettings(BaseModel):
    """Stage fields of the settings panel."""
    speed: Optional[StageSpeed] = None


class OpticsSettings(BaseModel):
    """Objective selection of the settings panel."""
    objective: Optional[str] = None


class HwSettingsRequest(BaseModel):
    """Body of POST /api/hw/settings. Omitted sections are left unchanged."""
    ccd: Optional[CcdSettings] = None
    laser: Optional[LaserSettings] = None
    camera: Optional[CameraSettings] = None
    stage: Optional[StageSettings] = None
    optics: Optional[OpticsSettings] = None


AcquireSpectrumRequest = public_args_model(acquire_spectrum, "AcquireSpectrumRequest")


class ExperimentRequest(BaseModel):
    """Body of the agent requests."""
    message: str
    session_id: Optional[str] = None
    agent: Optional[str] = None

