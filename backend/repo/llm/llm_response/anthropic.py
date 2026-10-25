"""Response types of the Anthropic Messages API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

STOP_KEYS = ("stop_reason",)
TRUNCATED = frozenset({"max_tokens"})
BLOCKED = frozenset({"refusal"})
SERVED_KEYS = ("model",)
REQUEST_ID_KEYS = ("id",)
TIMING_KEYS: tuple[str, ...] = ()
TOK_PER_S_BASIS = "wall"
THINKING_TOKEN_KEYS: tuple[str, ...] = ()
CACHE_KEYS = ("usage.cache_read_input_tokens",
              "usage.cache_creation_input_tokens")


class TextBlock(TypedDict, total=False):
    """Text content block."""
    type: str
    text: str


class ThinkingBlock(TypedDict, total=False):
    """Thinking content block."""
    type: str
    thinking: str
    signature: str


class ToolUseBlock(TypedDict, total=False):
    """Tool call content block."""
    type: str
    id: str
    name: str
    input: dict[str, Any]


class Usage(TypedDict, total=False):
    """Token usage."""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    inference_geo: str


class StopDetails(TypedDict, total=False):
    """Stop details, present only when stop_reason == "refusal"."""

    type: str
    category: str | None
    explanation: str


class Message(TypedDict, total=False):
    """Messages API response body."""
    id: str
    type: str
    role: str
    model: str
    content: list[TextBlock | ThinkingBlock | ToolUseBlock]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence",
                         "tool_use", "pause_turn", "refusal"]
    stop_sequence: str | None
    stop_details: StopDetails | None
    usage: Usage
