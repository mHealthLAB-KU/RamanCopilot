"""Instrument constants read from the vendor Config.ini, or from virtual_config.ini on virtual hardware."""
import configparser
import re
from dataclasses import dataclass
from pathlib import Path

from backend.common.config.env import env
from backend.common.config.virtual_hw import VIRTUAL_HW

CONFIG_PATH = (str(Path(__file__).with_name("virtual_config.ini")) if VIRTUAL_HW
               else r"C:\WEVE\Rays-ON_20260115\Config.ini")

_cfg = configparser.ConfigParser(strict=False)
_cfg.read(CONFIG_PATH, encoding="cp949")

STAGE_MAX_X    = _cfg.getfloat("STAGE_INFO", "MaxX")
STAGE_MAX_Y    = _cfg.getfloat("STAGE_INFO", "MaxY")
STAGE_CENTER_X = _cfg.getfloat("STAGE_INFO", "CenterX")
STAGE_CENTER_Y = _cfg.getfloat("STAGE_INFO", "CenterY")

STAGE_VELOCITY: tuple[float, float, float] = tuple(
    float(v) for v in _cfg.get("STAGE_INFO", "Velocity").split("/"))  # type: ignore[assignment]

CAMERA_WIDTH   = _cfg.getint("TUCSEN", "Width")
CAMERA_HEIGHT  = _cfg.getint("TUCSEN", "Height")
CAMERA_EXPOSURE_MS   = _cfg.getfloat("TUCSEN", "ExposureTime")
CAMERA_AUTO_EXPOSURE = _cfg.getboolean("TUCSEN", "AExpo", fallback=False)


_CALIB_BY_OBJECTIVE = {
    "20x": (1.002, 1.004),
}


@dataclass(frozen=True)
class Lens:
    """Objective lens entry from Config.ini."""
    objective: str
    name: str
    width_um: float
    height_um: float
    calib_x: float
    calib_y: float
    offset_x_mm: float
    offset_y_mm: float


def _read_lenses() -> dict[str, Lens]:
    lenses: dict[str, Lens] = {}
    for i in range(1, _cfg.getint("LENS_INFO", "TotalLens", fallback=1) + 1):
        sec = f"LENS_{i}"
        if not _cfg.has_section(sec):
            continue
        name = _cfg.get(sec, "Name", fallback="").strip()
        m = re.search(r"(\d+x)$", name.lower())
        if not m or m.group(1) in lenses:
            continue
        calib_x, calib_y = _CALIB_BY_OBJECTIVE.get(m.group(1), (1.0, 1.0))
        lenses[m.group(1)] = Lens(
            objective=m.group(1), name=name,
            width_um=_cfg.getfloat(sec, "Width"), height_um=_cfg.getfloat(sec, "Height"),
            calib_x=calib_x, calib_y=calib_y,
            offset_x_mm=_cfg.getfloat(sec, "OffsetX", fallback=0.0),
            offset_y_mm=_cfg.getfloat(sec, "OffsetY", fallback=0.0))
    return dict(sorted(lenses.items(), key=lambda kv: int(kv[0][:-1])))


LENSES: dict[str, Lens] = _read_lenses()

OBJECTIVE = env("RAMAN_OBJECTIVE", default="50x")

DEFAULT_OBJECTIVE = OBJECTIVE.strip().lower()
if DEFAULT_OBJECTIVE not in LENSES:
    raise ValueError(f"OBJECTIVE={OBJECTIVE!r} (RAMAN_OBJECTIVE) has no matching lens in Config.ini. "
                     f"Valid values: {list(LENSES)}")

CAMERA_SENSOR_BIN = 2  # camera sensor binning

_CCD_SECTION = "ANDOR_EM" if _cfg.getint("SYSTEM_INFO", "CCDType", fallback=0) == 1 else "ANDOR_IDUS"

_ACQ_MODES   = {1: "single", 2: "accumulate", 3: "kinetic", 5: "run_till_abort"}
_READ_MODES  = {0: "fvb", 3: "single_track", 4: "image"}
_TRIGGERS    = {0: "internal", 1: "external", 6: "external_start", 7: "external_exposure",
                9: "external_fvb_em", 10: "software"}
_SHUTTERS    = {0: "auto", 1: "open", 2: "close"}


def _code(table: dict, key: str):
    """Map a Config.ini numeric code to the tool-layer name. An unknown code stops startup — silently substituting a default would measure under different conditions than the vendor software."""
    raw = _cfg.getint(_CCD_SECTION, key)
    if raw not in table:
        raise ValueError(f"Config.ini [{_CCD_SECTION}] {key}={raw} is not supported. Valid values: {table}")
    return table[raw]


@dataclass(frozen=True)
class CcdStartup:
    """CCD acquisition settings applied at startup."""
    acq_mode: str
    exposure_s: float
    num_acc: int
    accumulation_cycle_s: float
    num_kin: int
    kinetic_cycle_s: float
    trigger: str
    read_mode: str
    track_center: int
    track_height: int
    vs_index: int
    hs_index: int
    preamp_index: int
    shutter: str
    target_temp: int
    cooler: bool
    cosmic_filter: bool
    hflip: bool


CCD_STARTUP = CcdStartup(
    acq_mode=_code(_ACQ_MODES, "AcquisitionMode"),
    exposure_s=_cfg.getfloat(_CCD_SECTION, "ExposureTime"),
    num_acc=_cfg.getint(_CCD_SECTION, "NumberOfAccumulation"),
    accumulation_cycle_s=_cfg.getfloat(_CCD_SECTION, "AccumulationCycle"),
    num_kin=_cfg.getint(_CCD_SECTION, "NumberOfKinetic"),
    kinetic_cycle_s=_cfg.getfloat(_CCD_SECTION, "KineticCycle"),
    trigger=_code(_TRIGGERS, "TriggerMode"),
    read_mode=_code(_READ_MODES, "ReadoutMode"),
    track_center=_cfg.getint(_CCD_SECTION, "TrackCentre", fallback=0),
    track_height=_cfg.getint(_CCD_SECTION, "TrackHeight", fallback=0),
    vs_index=_cfg.getint(_CCD_SECTION, "VSSpeedIndex"),
    hs_index=_cfg.getint(_CCD_SECTION, "ReadoutRateIndex"),
    preamp_index=_cfg.getint(_CCD_SECTION, "PreAmplifierGainIndex"),
    shutter=_code(_SHUTTERS, "ShutterMode"),
    target_temp=_cfg.getint(_CCD_SECTION, "TargetTemperature"),
    cooler=_cfg.getboolean(_CCD_SECTION, "EnableCooler"),
    cosmic_filter=_cfg.getboolean(_CCD_SECTION, "EnableCosmicRayFilter", fallback=False),
    hflip=_cfg.getboolean(_CCD_SECTION, "Reverse", fallback=False),
)

LASER_PORT = f"COM{_cfg.getint('MOTOR_INFO', 'Port')}"


@dataclass(frozen=True)
class Motor:
    """Motor entry from the MOTOR_INFO sections of Config.ini."""
    axis: str
    homing_vel: int
    moving_vel: int
    offset: int


def _read_motors() -> dict[str, Motor]:
    motors = {}
    for i in range(1, _cfg.getint("MOTOR_INFO", "TotalMotor", fallback=0) + 1):
        sec = f"MOTOR_{i}"
        if not _cfg.has_section(sec) or not _cfg.getboolean(sec, "Enable", fallback=False):
            continue
        motors[_cfg.get(sec, "MotorType").strip()] = Motor(
            axis=f"{_cfg.getint(sec, 'MotorNumber'):02d}",
            homing_vel=_cfg.getint(sec, "HomingVelocity"),
            moving_vel=_cfg.getint(sec, "MovingVelocity"),
            offset=_cfg.getint(sec, "Offset"))
    return motors


MOTORS: dict[str, Motor] = _read_motors()

ND_FILTER_CAL: list[tuple[float, int]] = sorted(
    ((float(k), int(v)) for k, v in _cfg.items("ND_FILTER") if re.fullmatch(r"\d+(\.\d+)?", k)),
    key=lambda kv: -kv[0])

BEAM_SPLITTER_POS: dict[str, int] = {k.lower(): _cfg.getint("BEAM_SPLITTER", k)
                                     for k in ("Image", "Signal", "Silicon")}

LASER_FILTER_POS = _cfg.getint("LASER_FILTER", "Calibration")

_sel_rayleigh   = _cfg.getint("AUTO_CALIBRATION", "SelectedRayleigh", fallback=0)
LASER_NM        = _cfg.getfloat("AUTO_CALIBRATION", f"RayleighWaveLength{_sel_rayleigh + 1}")
SI_PEAK_OFFSET = _cfg.getfloat("AUTO_CALIBRATION", f"SiPeakOffset{_sel_rayleigh + 1}")

_sel_type       = _cfg.get("AUTO_CALIBRATION", "SelectedType", fallback="0")
FOCAL_LENGTH_MM = _cfg.getfloat(f"TYPE-{_sel_type}", "FocalLength")
PIXEL_WIDTH_UM   = _cfg.getfloat("AUTO_CALIBRATION", "PixelWidth")
GROOVE_PER_MM    = _cfg.getfloat(f"TYPE-{_sel_type}", "Groove")
TILT_ANGLE_DEG   = _cfg.getfloat(f"TYPE-{_sel_type}", "TiltAngle")
PIXEL_COUNT      = _cfg.getint("AUTO_CALIBRATION", "PixelCount")

RAMAN_CENTER_CM1 = _cfg.getfloat(f"TYPE-{_sel_type}", "Center1")

RAMAN_PIXEL_OFFSET = -5.35  # px, single-point shift that puts the measured Si peak at 520.7 cm-1
STAGE_MIN_Z = -1.0  # mm, not in Config.ini
STAGE_MAX_Z =  1.0  # mm, not in Config.ini

LASER_FULL_MW = 45.7  # mW at 532 nm and 100% transmission, from the vendor test report
