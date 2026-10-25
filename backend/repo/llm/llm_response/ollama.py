"""Response types of the Ollama chat API."""

from __future__ import annotations

from typing import Any, TypedDict

STOP_KEYS = ("done_reason",)
TRUNCATED = frozenset({"length"})
BLOCKED: frozenset[str] = frozenset()
SERVED_KEYS = ("model",)
REQUEST_ID_KEYS: tuple[str, ...] = ()
TIMING_KEYS = ("total_duration",
               "load_duration",
               "prompt_eval_duration",
               "eval_duration")
TOK_PER_S_BASIS = "eval"
THINKING_TOKEN_KEYS: tuple[str, ...] = ()
CACHE_KEYS: tuple[str, ...] = ()


class ToolCallFunction(TypedDict, total=False):
    """Function name and arguments of a tool call."""
    name: str
    arguments: dict[str, Any]


class ToolCall(TypedDict, total=False):
    """One tool call."""
    function: ToolCallFunction


class Message(TypedDict, total=False):
    """Assistant message."""
    role: str
    content: str
    thinking: str
    tool_calls: list[ToolCall]
    images: list[str] | None


class ChatResponse(TypedDict, total=False):
    """/api/chat response body."""
    model: str
    created_at: str
    message: Message
    done: bool

    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class ShowResponse(TypedDict, total=False):
    """Fields of POST /api/show used here."""

    capabilities: list[str]
