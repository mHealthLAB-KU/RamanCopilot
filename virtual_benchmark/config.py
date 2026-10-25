"""Loads the virtual sample settings from config.json."""

from __future__ import annotations

import json
from pathlib import Path

from backend.common.config.hw import LASER_FULL_MW

_CFG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))

BEST_POWER_PCT = min(float(_CFG["best_laser_pow"]), LASER_FULL_MW) * 100.0 / LASER_FULL_MW  # best_laser_pow is in mW, clipped to the laser maximum
BEST_EXPOSURE_S = float(_CFG["best_laser_exp_s"])
TARGET_PEAK_CM1 = float(_CFG["target_peak_cm1"])
FWHM_MIN = float(_CFG["all_bands_FWHM"])

TARGET_AREA = float(_CFG["target_area"])

BANDS = {k: [tuple(float(x) for x in b) for b in v] for k, v in _CFG["bands"].items()}
SCALE = {k: float(v) for k, v in _CFG["scale"].items()}
COLORS_RGB = {k: tuple(float(x) for x in v) for k, v in _CFG["colors_rgb"].items()}

SPOT_W_UM = float(_CFG["spot_w_um"])
ALPHA = float(_CFG["damage"]["alpha"])
W_DECAY = float(_CFG["damage"]["w_decay"])
PEAK_MAX_COUNTS = float(_CFG["ccd"]["peak_max_counts"])
FULL_WELL = float(_CFG["ccd"]["full_well"])
READ_SIGMA = float(_CFG["ccd"]["read_sigma"])

MAP_W = int(_CFG["world"]["map_w"])
MAP_H = int(_CFG["world"]["map_h"])
FBM_BASE = int(_CFG["world"]["fbm_base"])
FBM_OCTAVES = int(_CFG["world"]["fbm_octaves"])
GAMMA = float(_CFG["world"]["gamma"])
REF_OBJECTIVE = str(_CFG["world"]["ref_objective"]).strip().lower()
