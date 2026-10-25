"""Virtual Andor CCD with the same interface as the real driver."""

from __future__ import annotations

import threading
import time

import numpy as np

from backend.common.config.hw import PIXEL_COUNT


class consts:  # Andor temperature status codes (atmcd), kept here so the virtual CCD needs no vendor SDK
    """Andor temperature status codes."""
    DRV_TEMP_OFF = 20034
    DRV_TEMP_NOT_STABILIZED = 20035
    DRV_TEMP_STABILIZED = 20036
    DRV_TEMP_NOT_REACHED = 20037
    DRV_TEMP_DRIFT = 20040


_NX = int(PIXEL_COUNT)
_NY = 256  # detector rows

_FULL_WELL = 65535  # counts
_DARK_LEVEL = 500.0  # counts, offset plus dark current
_READ_NOISE = 12.0  # counts

_AMBIENT_C = 20.0  # degC


def _synthesize(axis_cm1, color_bgr, ctx) -> np.ndarray:
    """Synthesize a 1D spectrum for the given Raman axis."""
    n = len(axis_cm1)
    rng = np.random.default_rng()

    signal = np.full(n, _DARK_LEVEL, dtype=np.float64)
    if ctx["laser_armed"]:
        signal += 40.0 * ctx["power_pct"] * ctx["exposure_s"] * ctx["num_acc"]

    signal += rng.normal(0.0, _READ_NOISE, n)
    signal += rng.normal(0.0, 1.0, n) * np.sqrt(np.maximum(signal, 0.0))

    return np.clip(signal, 0, _FULL_WELL).astype(np.int32)


class VirtualCCD:
    """Virtual Andor CCD that stores parameters and synthesizes spectra on acquisition."""

    TIME_SCALE = 1.0

    TEMP_RATE_C_PER_S = 20.0

    def __init__(self, debug: bool = False, initialize_to_defaults: bool = True,
                 mgr=None, scene=None):
        """Create a virtual CCD; mgr and scene tell it where the sample is being observed."""
        self._mgr = mgr
        self._scene = scene
        self._lock = threading.RLock()

        self.Nx, self.Ny = _NX, _NY
        self.Nx_ro, self.Ny_ro = _NX, 1
        self.hbin = 1
        self.ro_mode = 'FULL_VERTICAL_BINNING'
        self.em_mode = False

        self.exposure_time = 0.1
        self._kin_time = None
        self.accumulation_time = self.kinetic_cycle_time = 0.1
        self.num_acc = 1
        self.num_kin = 1
        self.aq_mode = 'single'
        self.trigger_mode = 'internal'
        self.shutter_explicit = False
        self.shutter_mode = 'close'

        self.numVSSpeeds = 4
        self.VSSpeeds = [4.25, 8.25, 16.25, 32.25]
        self.numHSSpeeds_Conventional = [4]
        self.HSSpeeds_Conventional = [[3.0, 1.0, 0.05, 0.01]]
        self.preamp_gains = [1.0, 2.0, 4.0]
        self.vs_speed_index = 0
        self.hs_speed_index = 0
        self.preamp_gain_i = 0
        self.ad_channel = 0
        self.output_amp = 1
        self.cosmic_ray_filter = False
        self.hflip, self.vflip = False, False

        self.temperature_setpoint = -40
        self.cooler_on = False
        self._temp_c = _AMBIENT_C
        self._temp_t0 = time.time()
        self.temperature_status_num = consts.DRV_TEMP_OFF

        self._acq_deadline = 0.0
        self._calibrator = None

    def _tick_temp(self) -> None:
        """Move the temperature toward the target (or ambient) for the elapsed time."""
        now = time.time()
        dt, self._temp_t0 = now - self._temp_t0, now
        target = float(self.temperature_setpoint) if self.cooler_on else _AMBIENT_C
        step = self.TEMP_RATE_C_PER_S * max(dt, 0.0)
        if abs(target - self._temp_c) <= step:
            self._temp_c = target
        else:
            self._temp_c += step * (1.0 if target > self._temp_c else -1.0)

        if not self.cooler_on:
            self.temperature_status_num = consts.DRV_TEMP_OFF
        elif abs(self._temp_c - target) < 0.5:
            self.temperature_status_num = consts.DRV_TEMP_STABILIZED
        else:
            self.temperature_status_num = consts.DRV_TEMP_NOT_REACHED

    def get_temperature(self) -> int:
        """Return the CCD temperature as an int, like the real driver."""
        with self._lock:
            self._tick_temp()
            return int(round(self._temp_c))

    temp_status_dict = {consts.DRV_TEMP_OFF: 'OFF',
                        consts.DRV_TEMP_NOT_STABILIZED: 'NOT_STABILIZED',
                        consts.DRV_TEMP_STABILIZED: 'STABILIZED',
                        consts.DRV_TEMP_NOT_REACHED: 'NOT_REACHED',
                        consts.DRV_TEMP_DRIFT: 'DRIFT'}

    def get_temperature_status(self) -> str:
        with self._lock:
            self._tick_temp()
            return self.temp_status_dict.get(self.temperature_status_num, 'UNKNOWN')

    def set_temperature(self, t):
        with self._lock:
            self._tick_temp()
            self.temperature_setpoint = int(t)
        return self.temperature_setpoint

    def set_cooler(self, on: bool):
        with self._lock:
            self._tick_temp()
            self.cooler_on = bool(on)
        return self.cooler_on

    def set_ad_channel(self, chan_i: int = 0):
        self.ad_channel = int(chan_i)

    def get_fastest_recommended_vs_speed(self):
        return (0, self.VSSpeeds[0])

    def set_vs_speed(self, index: int):
        self.vs_speed_index = int(index)

    def set_hs_speed_conventional(self, index: int):
        self.hs_speed_index = int(index)

    def set_preamp_gain(self, index: int):
        self.preamp_gain_i = int(index)

    def set_output_amp(self, amp: int):
        self.output_amp = int(amp)

    def has_em_ccd(self) -> bool:
        return False

    def set_cosmic_ray_filter(self, on: bool):
        self.cosmic_ray_filter = bool(on)

    def set_image_flip(self, hflip: bool = False, vflip: bool = False):
        self.hflip, self.vflip = bool(hflip), bool(vflip)

    def set_shutter_open(self, *a, **kw):
        self.shutter_mode = 'open'

    def set_shutter_close(self, *a, **kw):
        self.shutter_mode = 'close'

    def set_shutter_auto(self, *a, **kw):
        self.shutter_mode = 'auto'

    def set_ro_full_vertical_binning(self, hbin: int = 1):
        self.ro_mode, self.hbin = 'FULL_VERTICAL_BINNING', int(hbin)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), 1

    def set_ro_single_track(self, center, width: int = 1, hbin: int = 1):
        self.ro_mode, self.hbin = 'SINGLE_TRACK', int(hbin)
        self.single_track_center, self.single_track_width = int(center), int(width)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), 1

    def set_ro_image_mode(self, hbin: int = 1, vbin: int = 1, **kw):
        self.ro_mode, self.hbin = 'IMG', int(hbin)
        self.Nx_ro = int(self.Nx / self.hbin)
        self.Ny_ro = int(self.Ny / max(int(vbin), 1))

    def get_current_hbin(self) -> int:
        return self.hbin

    def set_exposure_time(self, dt):
        self.exposure_time = float(dt)
        self.get_acquisition_timings()
        return self.exposure_time

    def get_exposure_time(self):
        return self.exposure_time

    def get_acquisition_timings(self):
        """Fill the timing cache the same way as the real driver."""
        self.accumulation_time = self.exposure_time
        self.kinetic_cycle_time = max(self._kin_time or 0.0, self.exposure_time)
        return self.exposure_time, self.accumulation_time, self.kinetic_cycle_time

    def set_num_accumulations(self, num):
        self.num_acc = max(1, int(num))

    def set_num_kinetics(self, num):
        self.num_kin = max(1, int(num))

    def set_aq_single_scan(self, exposure=None):
        self.aq_mode = 'single'
        if exposure is not None:
            self.set_exposure_time(exposure)

    def set_aq_accumulate_scan(self, exposure_time=None, num_acc=None, cycle_time=None):
        self.aq_mode = 'accumulate'
        if exposure_time is not None:
            self.set_exposure_time(exposure_time)
        if num_acc is not None:
            self.set_num_accumulations(num_acc)

    def set_aq_kinetic_scan(self, exp_time=None, num_acc=None, acc_time=None,
                            num_kin=None, kin_time=None):
        self.aq_mode = 'kinetic'
        if exp_time is not None:
            self.set_exposure_time(exp_time)
        if num_acc is not None:
            self.set_num_accumulations(num_acc)
        if num_kin is not None:
            self.set_num_kinetics(num_kin)
        if kin_time is not None:
            self._kin_time = float(kin_time)
        self.get_acquisition_timings()

    def set_aq_run_till_abort_scan(self):
        self.aq_mode = 'run_till_abort'

    def set_trigger_mode(self, mode: str):
        self.trigger_mode = str(mode)

    def _acc_per_frame(self) -> int:
        """Return the number of exposures summed per frame (accumulation applies only in accumulate/kinetic mode)."""
        return max(int(self.num_acc), 1) if self.aq_mode in ('accumulate', 'kinetic') else 1

    def _acq_seconds(self) -> float:
        frames = self.num_kin if self.aq_mode == 'kinetic' else 1
        return self.exposure_time * self._acc_per_frame() * max(frames, 1) * self.TIME_SCALE

    def get_status(self) -> str:
        return 'ACQUIRING' if time.time() < self._acq_deadline else 'IDLE'

    def prepare_acquisition(self):
        return None

    def start_acquisition(self):
        """Start an acquisition asynchronously."""
        self._acq_deadline = time.time() + self._acq_seconds()

    def send_software_trigger(self):
        return None

    def abort_acquisition(self):
        self._acq_deadline = 0.0

    def free_internal_memory(self):
        return None

    def create_buffer(self):
        return None

    def _axis_cm1(self) -> np.ndarray:
        """Return the Raman shift axis (pixel index if no calibrator is attached)."""
        cal = self._calibrator
        if cal is None:
            return np.arange(self.Nx_ro, dtype=np.float64)
        return np.array([float(cal.pixel_to_raman_shift(p)) for p in range(self.Nx_ro)])

    def _ctx(self) -> dict:
        """Return what the laser is currently hitting on the sample."""
        laser = getattr(self._mgr, "laser", None)
        armed = bool(getattr(laser, "_power_set", False)) and bool(getattr(laser, "is_on", False))
        if self.shutter_mode == 'close':
            armed = False
        return {"exposure_s": float(self.exposure_time),
                "power_pct": float(getattr(laser, "power_pct", 0.0) or 0.0),
                "num_acc": self._acc_per_frame(),
                "laser_armed": armed}

    def _color(self) -> tuple[int, int, int]:
        stage = getattr(self._mgr, "stage", None)
        if self._scene is None or stage is None:
            return (0, 0, 0)
        pos = stage.get_position()
        return (0, 0, 0) if pos is None else self._scene.color_at(pos[0], pos[1])

    def _one_frame(self) -> np.ndarray:
        """Return one (Ny_ro, Nx_ro) frame, repeating the spectrum along the vertical axis."""
        line = _synthesize(self._axis_cm1(), self._color(), self._ctx())
        return np.tile(line, (max(self.Ny_ro, 1), 1))

    def get_acquired_data(self) -> np.ndarray:
        with self._lock:
            if self.aq_mode == 'kinetic':
                return np.stack([self._one_frame() for _ in range(max(self.num_kin, 1))])
            return self._one_frame()

    def start_acquisition_cycle(self, trigger_mode_str: str = 'internal',
                                timeout_ms=None) -> dict:
        """Acquire, attach calibration and return a dict (single/accumulate), taking the same time as the real CCD."""
        with self._lock:
            wait_s = self._acq_seconds()
            if timeout_ms is not None and wait_s * 1000.0 > float(timeout_ms):
                return {"intensity": [], "calibrated": False,
                        "error": "WaitForAcquisition failed (timeout or no trigger)"}
            if wait_s > 0:
                time.sleep(wait_s)
            self._acq_deadline = 0.0

            intensity = self._one_frame().flatten().tolist()
            result = {"intensity": intensity, "calibrated": False}
            cal = self._calibrator
            if cal is not None:
                pixels = range(len(intensity))
                result.update({
                    "calibrated": True,
                    "raman_shift_cm-1": [float(cal.pixel_to_raman_shift(p)) for p in pixels],
                    "wavelength_nm": [float(cal.pixel_to_wavelength(p)) for p in pixels],
                    "laser_nm": float(cal.laser_nm),
                })
            return result

    def close(self):
        self.cooler_on = False
        self._acq_deadline = 0.0

    def __repr__(self) -> str:
        return (f"<VirtualCCD {self.aq_mode}/{self.ro_mode} {self.Ny_ro}x{self.Nx_ro} "
                f"exp={self.exposure_time}s T={int(round(self._temp_c))}C>")
