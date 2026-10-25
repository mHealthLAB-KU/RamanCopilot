"""Parses a model string into provider and model name."""

from __future__ import annotations

PROVIDERS = ("ollama", "openai", "anthropic", "google")

DEFAULT_PROVIDER = "ollama"

PIP_PACKAGE = {
    "ollama": "langchain-ollama",
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google": "langchain-google-genai",
}


def parse_model(spec: str) -> tuple[str, str]:
    """Split "<provider>:<model>" into (provider, model); no registered prefix means ollama."""
    raw = str(spec or "").strip()
    head, sep, tail = raw.partition(":")
    if sep and head.strip().lower() in PROVIDERS:
        return head.strip().lower(), tail.strip()
    return DEFAULT_PROVIDER, raw


def describe(spec: str) -> str:
    """Return a one-line summary of provider and model."""
    provider, name = parse_model(spec)
    return f"{provider}:{name}"


__all__ = ["PROVIDERS", "DEFAULT_PROVIDER", "PIP_PACKAGE", "parse_model", "describe"]
