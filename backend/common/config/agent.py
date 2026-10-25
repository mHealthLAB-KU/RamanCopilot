"""Agent settings read from environment variables."""
from __future__ import annotations

from backend.common.config.env import env, flag

AGENT_ARCH = env("RAMAN_AGENT", default="ReAct")

CHAT_SESSION_ISOLATED = flag("RAMAN_CHAT_ISOLATED", default="1")

BATCH_GATE = flag("RAMAN_BATCH_GATE", default="1")

GRID_SCAN = flag("RAMAN_GRID_SCAN", default="0")

SPECTRUM_IMAGE = flag("RAMAN_SPECTRUM_IMAGE", default="1")
SPECTRUM_IMAGE_MAX_SIDE = 640  # px, longest side of the spectrum image sent to the model
SPECTRUM_IMAGE_JPEG_QUALITY = 75  # JPEG quality of that image
