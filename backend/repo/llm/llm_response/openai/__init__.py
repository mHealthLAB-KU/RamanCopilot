"""Response types of the OpenAI APIs, split by model family."""

from __future__ import annotations

MODULE = {
    "gpt-5.4": "gpt_5",
    "gpt-4o": "gpt_4o",
}

API_PATH = {
    "gpt-5.4": "responses",
    "gpt-4o": "chat_completions",
}

TRUNCATED = frozenset({"length",
                       "max_output_tokens"})
BLOCKED = frozenset({"content_filter"})
SERVED_KEYS = ("model",)
TIMING_KEYS: tuple[str, ...] = ()
TOK_PER_S_BASIS = "wall"
