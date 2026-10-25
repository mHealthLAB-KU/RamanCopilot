"""LLM settings: model, host, context size, token limit and timeout."""
from __future__ import annotations

from backend.common.config.env import env

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "ollama": "",
}


OLLAMA_MODEL = env("RAMAN_OLLAMA_MODEL", default="anthropic:claude-sonnet-4-6")

API_BASE_URL = env("RAMAN_API_BASE", default="")

MAX_TOKENS = int(env("RAMAN_MAX_TOKENS", default="0"))

OLLAMA_HOST = env("RAMAN_OLLAMA_HOST", "OLLAMA_HOST",
                  default="http://localhost:11434")

NUM_CTX = int(env("RAMAN_NUM_CTX", default="100000"))

LLM_TIMEOUT_S = float(env("RAMAN_LLM_TIMEOUT_S", default="3600"))
