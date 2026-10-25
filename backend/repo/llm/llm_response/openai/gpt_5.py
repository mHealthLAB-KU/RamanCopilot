"""Response types of the OpenAI Responses API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

STOP_KEYS = ("incomplete_details.reason",
             "status")
TRUNCATED = frozenset({"max_output_tokens"})
BLOCKED = frozenset({"content_filter"})
SERVED_KEYS = ("model",)
REQUEST_ID_KEYS = ("id",)
TIMING_KEYS: tuple[str, ...] = ()
TOK_PER_S_BASIS = "wall"
THINKING_TOKEN_KEYS = ("usage.output_tokens_details.reasoning_tokens",)
CACHE_KEYS = ("usage.input_tokens_details.cached_tokens",)


class InputTokensDetails(TypedDict, total=False):
    """Input token details."""
    cached_tokens: int


class OutputTokensDetails(TypedDict, total=False):
    """Output token details."""
    reasoning_tokens: int


class Usage(TypedDict, total=False):
    """Token usage (input/output naming of the Responses API)."""

    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class ReasoningSummaryPart(TypedDict, total=False):
    """Part of a reasoning summary."""
    type: str
    text: str


class ReasoningItem(TypedDict, total=False):
    """Reasoning item in output[]; summary is filled only when reasoning.summary was requested."""

    type: str
    id: str
    summary: list[ReasoningSummaryPart]
    encrypted_content: str


class OutputTextPart(TypedDict, total=False):
    """Text part of an output message."""
    type: str
    text: str
    annotations: list[dict[str, Any]]


class MessageItem(TypedDict, total=False):
    """Output message item."""
    type: str
    id: str
    role: str
    content: list[OutputTextPart]


class FunctionCallItem(TypedDict, total=False):
    """Function call output item."""
    type: str
    id: str
    call_id: str
    name: str
    arguments: str


class IncompleteDetails(TypedDict, total=False):
    """Reason an output is incomplete."""
    reason: Literal["max_output_tokens", "content_filter"]


class Response(TypedDict, total=False):
    """Responses API response body."""
    id: str
    object: str
    created_at: int
    model: str
    status: Literal["completed", "incomplete", "failed",
                    "in_progress", "cancelled", "queued"]
    incomplete_details: IncompleteDetails | None
    error: dict[str, Any] | None
    output: list[ReasoningItem | MessageItem | FunctionCallItem]
    usage: Usage
    service_tier: str
