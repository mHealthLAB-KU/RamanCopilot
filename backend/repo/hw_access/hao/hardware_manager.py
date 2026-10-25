"""Startup, shutdown and connection locks for the stage, CCD, camera and laser."""
from __future__ import annotations

import contextlib
import ctypes
import functools
import signal
import sys
import threading
import time
from pathlib import Path

_BACKEND      = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.repo.hw_access.hao.USE_stage_test import TangoController  # noqa: E402
from backend.repo.hw_access.hao.USE_laser_with_power import LaserController  # noqa: E402
from backend.repo.hw_access import hw_core  # noqa: E402
from backend.common import optics_map  # noqa: E402
from backend.repo.store import aim_store  # noqa: E402
from backend.common.config.hw import (  # noqa: E402
    CAMERA_AUTO_EXPOSURE, CAMERA_EXPOSURE_MS, CCD_STARTUP, LASER_PORT, STAGE_VELOCITY,
    STAGE_MAX_X, STAGE_MAX_Y, STAGE_CENTER_X, STAGE_CENTER_Y,
    LASER_NM, FOCAL_LENGTH_MM, RAMAN_CENTER_CM1,
    PIXEL_COUNT, PIXEL_WIDTH_UM, GROOVE_PER_MM, TILT_ANGLE_DEG, SI_PEAK_OFFSET, RAMAN_PIXEL_OFFSET,
)

try:
    from backend.repo.hw_access.SDKs.andor_codes import AndorCCD  # noqa: E402
except ImportError:
    AndorCCD = None

from backend.repo.hw_access.raman_calibration import RamanCalibrator  # noqa: E402

STAGE_DLL_PATH = str(Path(__file__).resolve().parent.parent / "SDKs" / "Tango_DLL.dll")


from backend.common.config import llm as cfg  # noqa: E402
from backend.common.config import virtual_hw as vhw_cfg  # noqa: E402

CCD_WARM_TARGET = -5  # degC, shutdown waits until the CCD is at or above this

DRV_SUCCESS             = 20002
DRV_TEMP_OFF            = 20034
DRV_TEMP_STABILIZED     = 20036
DRV_TEMP_NOT_REACHED    = 20037
DRV_TEMP_NOT_STABILIZED = 20035
DRV_TEMP_DRIFT          = 20040

INIT_SPEED_MM_S = 10.0  # stage speed for the startup move to the center

DEFAULT_OUTPUT_AMP = 1  # Andor output amplifier index


def _guarded(component: str):
    """Decorator that serializes an _init_<component>() call and prevents duplicate initialization."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            with self.component_lock(component):
                if getattr(self, component) is not None:
                    print(f"[SKIP]  {component} is already connected "
                          f"- skipping initialization (reconnecting without release would orphan the resource).")
                    return
                return fn(self, *args, **kwargs)
        return wrapper
    return deco


def _progress_line():
    """Return a one-line progress printer that redraws only when the text changes."""
    last = ["", 0]

    def draw(text: str) -> None:
        if text == last[0]:
            return
        print("\r" + text + " " * max(0, last[1] - len(text)), end="", flush=True)
        last[0], last[1] = text, len(text)

    return draw


def restore_laser_aim() -> None:
    """Restore the stored laser spot position for the current objective.

    The image center is used if the objective has never been measured.
    """
    saved = None if vhw_cfg.VIRTUAL_HW else aim_store.load(optics_map.objective())
    if saved is None:
        optics_map.reset_aim_px()
    else:
        optics_map.set_aim_px(*saved)
        print(f"[CAM]   laser spot restored ({optics_map.objective()}): ({saved[0]:.1f}, {saved[1]:.1f}) px")


class HardwareManager:
    """Manage the connection lifecycle of all instruments."""

    def __init__(self):
        self.stage  = None
        self.ccd    = None
        self.camera = None
        self.laser  = None
        self.ollama = None

        self._shutdown_called = False
        self._lock = threading.Lock()

        self._comp_locks: dict[str, threading.RLock] = {
            name: threading.RLock()
            for name in ("stage", "ccd", "camera", "laser", "ollama")
        }

        self._prev_signal_handlers: dict = {}
        if threading.current_thread() is threading.main_thread():
            for _sig in (signal.SIGINT, signal.SIGTERM):
                self._prev_signal_handlers[_sig] = signal.signal(_sig, self._signal_handler)
        else:
            print("[WARN] HardwareManager created in a worker thread - no signal handler registered")


    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C / SIGTERM by starting shutdown and returning immediately."""
        print("\n\n[!] shutdown signal received - shutting down safely after the CCD warms up (do not force quit)")
        threading.Thread(target=self.shutdown, name="hw-shutdown", daemon=False).start()

        prev = self._prev_signal_handlers.get(sig)
        if callable(prev):
            prev(sig, frame)
        else:
            raise KeyboardInterrupt


    def component_lock(self, component: str) -> threading.RLock:
        """Return the lock that serializes connect and disconnect of one instrument.

        Args:
            component (str): One of stage, ccd, camera or laser.

        Returns:
            threading.RLock: The lock for that instrument.

        Raises:
            KeyError: If the component name is unknown.
        """
        try:
            return self._comp_locks[component]
        except KeyError:
            raise ValueError(
                f"unknown component '{component}' "
                f"(expected one of {sorted(self._comp_locks)})"
            ) from None

    @contextlib.contextmanager
    def _shutdown_guard(self, component: str, timeout: float = 10.0):
        """Wait briefly for an instrument lock during shutdown, then proceed anyway."""
        acquired = self._comp_locks[component].acquire(timeout=timeout)
        if not acquired:
            print(f"[{component.upper()}] warning: another thread holds this instrument "
                  f"(gave up after {timeout}s) - proceeding with release.")
        try:
            yield
        finally:
            if acquired:
                self._comp_locks[component].release()


    @_guarded("stage")
    def _init_stage(self):
        """Connect and initialize the Tango stage, then move it to the center."""
        print("\n[STAGE] connecting stage...")
        if vhw_cfg.VIRTUAL_HW:
            from backend.repo.hw_access.hao_vertual.virtual_stage import VirtualStage
            tango = VirtualStage(STAGE_DLL_PATH)
        else:
            tango = TangoController(STAGE_DLL_PATH)

        if not tango.load_dll():
            raise RuntimeError("stage DLL load failed")
        if not tango.create_session():
            raise RuntimeError("stage session creation failed")
        if not tango.connect():
            raise RuntimeError("stage connection failed (check cable/power/driver)")

        v = INIT_SPEED_MM_S
        tango.set_velocity(v, v, v * 0.1, v)

        print("[STAGE] initialization moves: (0,0) -> (max,max) -> center")
        pos = tango.get_position()
        if pos is None:
            raise RuntimeError("stage position query failed - get_position() returned nothing")
        z = pos[2]

        tango.move_absolute(STAGE_CENTER_X, STAGE_CENTER_Y, z, wait=True)
        pos = tango.get_position()
        if pos is not None:
            print(f"        -> center      reached  X={pos[0]:.4f}  Y={pos[1]:.4f}")
        time.sleep(1)

        vx, vy, vz = STAGE_VELOCITY
        tango.set_velocity(vx, vy, vz, vx)
        print("[STAGE] initialization complete")
        self.stage = tango

    @_guarded("ccd")
    def _init_ccd(self):
        """Initialize the Andor CCD with factory calibration and start cooling."""
        print("\n[CCD]   connecting Andor CCD...")

        calibrator = None
        if RamanCalibrator is not None:
            calibrator = RamanCalibrator.from_factory_calibration(
                laser_nm=LASER_NM,
                f_mm=FOCAL_LENGTH_MM,
                raman_center_cm1=RAMAN_CENTER_CM1,
                pixel_count=PIXEL_COUNT,
                pixel_width_um=PIXEL_WIDTH_UM,
                groove=GROOVE_PER_MM,
                tilt_angle_deg=TILT_ANGLE_DEG,
                si_peak_offset=SI_PEAK_OFFSET,
                pixel_offset=RAMAN_PIXEL_OFFSET,
            )
            print(f"[CCD]   factory calibration applied: "
                  f"{calibrator._lut.min():.0f}~{calibrator._lut.max():.0f} cm-1")

        if vhw_cfg.VIRTUAL_HW:
            from virtual_benchmark import bands
            calibrator = bands.calibrator()
            print(f"[CCD]   replaced with the virtual benchmark axis: "
                  f"{calibrator._lut.min():.0f}~{calibrator._lut.max():.0f} cm-1 "
                  f"(grating {bands.GRATING_GROOVE:.0f} g/mm, center {bands.GRATING_CENTER_CM1:.0f}, "
                  f"laser {bands.LASER_NM:.3f} nm)")

        if vhw_cfg.VIRTUAL_HW:
            from backend.repo.hw_access.hao_vertual.virtual_ccd import VirtualCCD
            from virtual_benchmark.rig import attach_ccd, get_sample
            sample = get_sample()
            ccd = VirtualCCD(initialize_to_defaults=False, mgr=self, scene=sample)
            attach_ccd(ccd, self, sample)
        elif AndorCCD is None:
            raise RuntimeError("Andor SDK (pyAndorSDK2) not found; cannot initialize the CCD. "
                               "To run without instruments, start with RAMAN_VIRTUAL_HW=1.")
        else:
            ccd = AndorCCD(initialize_to_defaults=False)
        ccd._calibrator = calibrator

        st = CCD_STARTUP
        ccd.set_ad_channel(0)
        if ccd.numVSSpeeds > 0:
            ccd.set_vs_speed(st.vs_index)
        if ccd.numHSSpeeds_Conventional and ccd.numHSSpeeds_Conventional[0] > 0:
            ccd.set_hs_speed_conventional(st.hs_index)
        ccd.set_preamp_gain(st.preamp_index)
        if ccd.em_mode:
            ccd.set_output_amp(DEFAULT_OUTPUT_AMP)

        if st.acq_mode == "accumulate":
            ccd.set_aq_accumulate_scan(exposure_time=st.exposure_s, num_acc=st.num_acc,
                                       cycle_time=st.accumulation_cycle_s)
        elif st.acq_mode == "kinetic":
            ccd.set_aq_kinetic_scan(exp_time=st.exposure_s, num_acc=st.num_acc,
                                    num_kin=st.num_kin, kin_time=st.kinetic_cycle_s)
        elif st.acq_mode == "run_till_abort":
            ccd.set_aq_run_till_abort_scan()
            ccd.set_exposure_time(st.exposure_s)
        else:
            ccd.set_aq_single_scan(exposure=st.exposure_s)
        ccd.set_num_accumulations(st.num_acc)
        ccd.set_num_kinetics(st.num_kin)

        if st.read_mode == "single_track" and st.track_center > 0:
            ccd.set_ro_single_track(st.track_center, max(1, st.track_height))
        elif st.read_mode == "image":
            ccd.set_ro_image_mode()
        else:
            if st.read_mode == "single_track":
                print("[CCD]   Config.ini specifies single track but TrackCentre=0; using FVB")
            ccd.set_ro_full_vertical_binning()

        ccd.set_trigger_mode(st.trigger)
        {"auto": ccd.set_shutter_auto, "open": ccd.set_shutter_open,
         "close": ccd.set_shutter_close}[st.shutter]()
        ccd.set_cosmic_ray_filter(st.cosmic_filter and st.acq_mode == "accumulate")
        ccd.set_image_flip(hflip=st.hflip, vflip=False)
        print(f"[CCD]   Config.ini applied: {st.acq_mode} {st.exposure_s}s ×{st.num_acc} · {st.read_mode} · "
              f"trigger {st.trigger} · shutter {st.shutter} · VS#{st.vs_index} HS#{st.hs_index} "
              f"gain#{st.preamp_index}")

        ccd.get_acquisition_timings()

        ccd.set_temperature(st.target_temp)
        ccd.set_cooler(st.cooler)

        self.ccd = ccd
        print(f"[CCD]   connected - cooling {'started' if st.cooler else 'off'} (target {st.target_temp}°C). Measurements possible before reaching it")

    @_guarded("camera")
    def _init_camera(self):
        """Connect the TUCam camera and start streaming."""
        print("\n[CAM]   connecting TUCam camera...")
        if vhw_cfg.VIRTUAL_HW:
            from backend.repo.hw_access.hao_vertual.virtual_camera import VirtualCamera
            from virtual_benchmark.rig import get_sample
            cam = VirtualCamera(self, scene=get_sample(), exposure_ms=CAMERA_EXPOSURE_MS)
        else:
            from backend.repo.hw_access.hao.USE_camera_stream import StreamingTUCam
            cam = StreamingTUCam(exposure_ms=CAMERA_EXPOSURE_MS)
        if CAMERA_AUTO_EXPOSURE:
            cam.set_auto_exposure(True)
        restore_laser_aim()
        cam.start_stream()
        print("[CAM]   connected")
        self.camera = cam

    @_guarded("laser")
    def _init_laser(self):
        """Connect the laser controller."""
        print(f"\n[LASER] connecting laser... (port: {LASER_PORT})")
        if vhw_cfg.VIRTUAL_HW:
            from backend.repo.hw_access.hao_vertual.virtual_laser import VirtualLaser
            laser = VirtualLaser(port=LASER_PORT)
        else:
            laser = LaserController(port=LASER_PORT)
        if not (laser.ser and laser.ser.is_open):
            raise RuntimeError(f"laser port connection failed ({LASER_PORT})")
        print("[LASER] connected")
        self.laser = laser

    @_guarded("ollama")
    def _init_ollama(self):
        """Check the Ollama API connection (only when the model is served by Ollama)."""
        from backend.repo.llm import providers
        if providers.parse_model(cfg.OLLAMA_MODEL)[0] != "ollama":
            return
        import ollama

        print(f"\n[LLM]   checking Ollama connection... ({cfg.OLLAMA_HOST})")
        client = ollama.Client(host=cfg.OLLAMA_HOST)
        try:
            client.list()
        except Exception as e:
            raise RuntimeError(f"Ollama connection failed: {e}")
        print(f"[LLM]   connected (model: {cfg.OLLAMA_MODEL})")
        self.ollama = client


    def startup(self):
        """Initialize all instruments.

        The stage and the CCD are initialized first. The camera, laser and Ollama follow once both are ready.
        """
        print("=" * 60)
        print("  System initialization started")
        print("=" * 60)

        stage_exc: list[Exception | None] = [None]
        ccd_exc:   list[Exception | None] = [None]
        stage_done = threading.Event()
        ccd_done   = threading.Event()

        def _run_stage():
            try:
                self._init_stage()
            except Exception as e:
                stage_exc[0] = e
            finally:
                stage_done.set()

        def _run_ccd():
            try:
                self._init_ccd()
            except Exception as e:
                ccd_exc[0] = e
            finally:
                ccd_done.set()

        t_stage = threading.Thread(target=_run_stage, daemon=True, name="init-stage")
        t_ccd   = threading.Thread(target=_run_ccd,   daemon=True, name="init-ccd")
        t_stage.start()
        t_ccd.start()

        stage_done.wait()
        ccd_done.wait()

        if stage_exc[0] is not None:
            print(f"[WARN] [STAGE] initialization failed (skipped): {stage_exc[0]}")
        if ccd_exc[0] is not None:
            print(f"[WARN] [CCD]   initialization failed (skipped): {ccd_exc[0]}")

        print("\n" + "-" * 60)
        print("  [Phase 1 complete] -> Phase 2: camera / laser / LLM")
        print("-" * 60)

        for name, init_fn in [
            ("CAM",   self._init_camera),
            ("LASER", self._init_laser),
            ("LLM",   self._init_ollama),
        ]:
            try:
                init_fn()
            except Exception as e:
                print(f"[WARN] [{name}]   initialization failed (skipped): {e}")

        connected = [
            k for k, v in [
                ("STAGE", self.stage), ("CCD", self.ccd),
                ("CAM",   self.camera), ("LASER", self.laser),
                ("LLM",   self.ollama),
            ] if v is not None
        ]
        print("\n" + "=" * 60)
        print(f"  Initialization complete - connected: {', '.join(connected) if connected else 'none'}")
        print("=" * 60 + "\n")

        hw_core.init_hardware(
            stage=self.stage, 
            laser=self.laser, 
            ccd=self.ccd, 
            camera=self.camera
        )

    def shutdown(self):
        """Shut down all instruments safely.

        Blocks until the CCD has warmed up to CCD_WARM_TARGET, then releases the laser, camera and stage.
        """
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        print("\n" + "=" * 60)
        print("  System shutdown started")
        print("=" * 60)


        with self._shutdown_guard("ccd"):
            self._ccd_warmup_and_shutdown()

        with self._shutdown_guard("laser"):
            if self.laser is not None:
                try:
                    self.laser.laser_off()
                    if self.laser.ser and self.laser.ser.is_open:
                        self.laser.ser.close()
                    print("[LASER] disconnected")
                except Exception as e:
                    print(f"[LASER] exception during release: {e}")

        with self._shutdown_guard("camera"):
            if self.camera is not None:
                try:
                    self.camera.stop_stream()
                    self.camera.close()
                    print("[CAM]   disconnected")
                except Exception as e:
                    print(f"[CAM]   exception during release: {e}")

        with self._shutdown_guard("stage"):
            if self.stage is not None:
                try:
                    self.stage.disconnect()
                    self.stage.free_session()
                    print("[STAGE] disconnected")
                except Exception as e:
                    print(f"[STAGE] exception during release: {e}")

        print("\n" + "=" * 60)
        print("  All hardware disconnected")
        print("=" * 60 + "\n")

    def _ccd_warmup_and_shutdown(self):
        """Turn the cooler off, block until the CCD reaches CCD_WARM_TARGET, then shut down the SDK."""
        if self.ccd is None:
            return

        print(f"[CCD]   controlled warm-up to {CCD_WARM_TARGET}°C, then cooler OFF")
        print("        (the process will not exit before this step completes)")

        try:
            if self.ccd.get_status() == 'ACQUIRING':
                self.ccd.sdk.AbortAcquisition()
                time.sleep(0.5)
        except Exception:
            pass

        try:
            self.ccd.set_temperature(CCD_WARM_TARGET)
        except Exception as e:
            print(f"[CCD]   [WARN] exception setting the temperature target: {e}")

        _LABELS = {
            DRV_TEMP_STABILIZED:     "stabilized",
            DRV_TEMP_NOT_REACHED:    "cooling",
            DRV_TEMP_NOT_STABILIZED: "not stabilized",
            DRV_TEMP_DRIFT:          "drift",
            DRV_TEMP_OFF:            "cooler OFF",
        }
        draw = _progress_line()
        while True:
            try:
                t      = self.ccd.get_temperature()
                status = self.ccd.temperature_status_num
                label  = _LABELS.get(status, f"status={status}")

                draw(f"        warming up: {t:5d}°C  [{label}]  (target: >= {CCD_WARM_TARGET}°C)")

                if t >= CCD_WARM_TARGET:
                    print(f"\n[CCD]   warm-up complete: {t}°C")
                    break
            except Exception as e:
                print(f"\n[CCD]   [WARN] temperature read error: {e} - retrying...")

            time.sleep(5)

        try:
            self.ccd.set_cooler(False)
        except Exception as e:
            print(f"[CCD]   [WARN] CoolerOFF exception: {e}")

        try:
            self.ccd.close()
            print("[CCD]   Andor SDK shut down")
        except Exception as e:
            print(f"[CCD]   [WARN] SDK shutdown exception: {e}")


    def __enter__(self) -> "HardwareManager":
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


_manager: HardwareManager | None = None


def get_manager() -> HardwareManager:
    """Return the HardwareManager singleton.

    The caller is responsible for calling startup().

    Returns:
        HardwareManager: The shared manager.
    """
    global _manager
    if _manager is None:
        _manager = HardwareManager()
    return _manager
