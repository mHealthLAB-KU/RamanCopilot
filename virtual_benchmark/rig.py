"""Connects the virtual sample to the virtual instruments."""

from __future__ import annotations

import numpy as np

from backend.repo.hw_access.hao_vertual import virtual_ccd as _vccd
from backend.repo.hw_access.hao_vertual.virtual_camera import VirtualCamera
from backend.repo.hw_access.hao_vertual.virtual_ccd import VirtualCCD
from backend.repo.hw_access.hao_vertual.virtual_laser import VirtualLaser
from backend.repo.hw_access.hao_vertual.virtual_stage import VirtualStage
from backend.repo.hw_access import hw_core
from virtual_benchmark import bands, session_log
from virtual_benchmark.config import TARGET_AREA
from virtual_benchmark.sample import FULL_WELL, READ_SIGMA, VirtualSample


_SAMPLE: VirtualSample | None = None
_SAMPLE_KEY: tuple | None = None


def get_sample(seed: int | None = None, target_area: float = TARGET_AREA) -> VirtualSample:
    """Return the sample singleton shared by the camera and CCD."""
    global _SAMPLE, _SAMPLE_KEY
    from backend.common.config import virtual_hw as vhw_cfg
    seed = vhw_cfg.VBENCH_SEED if seed is None else seed
    key = (int(seed), float(target_area))
    if _SAMPLE is None or _SAMPLE_KEY != key:
        _SAMPLE = VirtualSample(int(seed), float(target_area))
        _SAMPLE_KEY = key
        print(f"[VBENCH] {_SAMPLE}")
    return _SAMPLE


def attach_ccd(ccd, mgr, sample: VirtualSample) -> None:
    """Make the virtual CCD synthesize spectra from the virtual sample."""
    ccd._calibrator = bands.calibrator()

    def _synthesize(axis_cm1, color_bgr, ctx):
        n = len(axis_cm1)
        exposure_s = float(ctx["exposure_s"])
        n_acc = max(int(ctx["num_acc"]), 1)

        if not ctx["laser_armed"]:
            out = np.random.default_rng().normal(0.0, READ_SIGMA * np.sqrt(n_acc), n)
            return np.clip(out, 0, FULL_WELL * n_acc).astype(np.int32)

        stage = getattr(mgr, "stage", None)
        pos = None if stage is None else stage.get_position()
        x, y = (sample.center_mm if pos is None else (pos[0], pos[1]))

        total = np.zeros(n, dtype=np.float64)
        for _ in range(n_acc):
            total += sample.measure(x, y, float(ctx["power_pct"]), exposure_s,
                                    axis_cm1=np.asarray(axis_cm1, dtype=np.float64))
        out = np.clip(total, 0, FULL_WELL * n_acc).astype(np.int32)
        session_log.record(axis_cm1, out, ctx, x, y)
        return out

    _vccd._synthesize = _synthesize


class VirtualRig:
    """Four virtual instruments and one sample, acting as the hardware manager."""

    def __init__(self, seed: int | None = None, target_area: float = TARGET_AREA):
        self.sample = get_sample(seed, target_area)

        self.stage = VirtualStage()
        self.laser = VirtualLaser()
        self.camera = VirtualCamera(self, scene=self.sample)
        self.ccd = VirtualCCD(mgr=self, scene=self.sample)
        attach_ccd(self.ccd, self, self.sample)

        self.stage.load_dll()
        self.stage.create_session()
        self.stage.connect()
        self.camera.start_stream()


    def install(self) -> "VirtualRig":
        """Install this rig into the tool layer."""
        hw_core.init_hardware(stage=self.stage, laser=self.laser,
                              ccd=self.ccd, camera=self.camera)
        return self


    def truth_at(self, x_mm=None, y_mm=None) -> dict:
        """Return the ground-truth state at a position."""
        if x_mm is None:
            pos = self.stage.get_position()
            x_mm, y_mm = (self.sample.center_mm if pos is None else (pos[0], pos[1]))
        f = self.sample.composition_at(x_mm, y_mm)
        return {"x_mm": float(x_mm), "y_mm": float(y_mm),
                "f_bg": float(f[0]), "f_tgt2": float(f[1]), "f_tgt": float(f[2]),
                "dose": self.sample.dose_at(x_mm, y_mm),
                "survival": self.sample.survival_at(x_mm, y_mm)}

    def burned_spots(self) -> int:
        """Return the number of burned spots."""
        return self.sample.burned_spots()

    def __repr__(self) -> str:
        pos = self.stage.get_position()
        where = "?" if pos is None else f"({pos[0]:.3f}, {pos[1]:.3f})"
        return f"<VirtualRig seed={self.sample.world.seed} stage={where} burned={self.burned_spots()}>"


def install(seed: int | None = None, target_area: float = TARGET_AREA) -> VirtualRig:
    """Create a rig and install it into the tool layer."""
    return VirtualRig(seed, target_area).install()
