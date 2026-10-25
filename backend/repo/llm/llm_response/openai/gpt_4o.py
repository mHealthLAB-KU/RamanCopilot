"""Response types of the OpenAI Chat Completions API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

STOP_KEYS = ("finish_reason",)
TRUNCATED = frozenset({"length"})
BLOCKED = frozenset({"content_filter"})
SERVED_KEYS = ("model",)
REQUEST_ID_KEYS = ("id", "system_fingerprint")
TIMING_KEYS: tuple[str, ...] = ()
TOK_PER_S_BASIS = "wall"
THINKING_TOKEN_KEYS = ("usage.completion_tokens_details.reasoning_tokens",)
CACHE_KEYS = ("usage.prompt_tokens_details.cached_tokens",)


class PromptTokensDetails(TypedDict, total=False):
    """Prompt token details."""
    cached_tokens: int


class CompletionTokensDetails(TypedDict, total=False):
    """Completion token details."""
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int
    audio_tokens: int


class Usage(TypedDict, total=False):
    """Token usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class Message(TypedDict, total=False):
    """Assistant message."""
    role: str
    content: str | None
    refusal: str | None
    tool_calls: list[dict[str, Any]]
    annotations: list[dict[str, Any]]


class Choice(TypedDict, total=False):
    """One completion choice."""
    index: int
    message: Message
    logprobs: dict[str, Any] | None
    finish_reason: Literal["stop", "length", "content_filter",
                           "tool_calls", "function_call"]


class ChatCompletion(TypedDict, total=False):
    """Chat Completions response body."""
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    service_tier: str
    system_fingerprint: str
