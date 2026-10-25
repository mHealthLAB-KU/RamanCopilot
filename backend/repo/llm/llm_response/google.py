"""Response types of the Google Gemini generateContent API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

STOP_KEYS = ("candidates[].finishReason",)
TRUNCATED = frozenset({"MAX_TOKENS"})
BLOCKED = frozenset({"SAFETY",
                     "RECITATION",
                     "PROHIBITED_CONTENT",
                     "BLOCKLIST"})
PROMPT_BLOCKED = frozenset({"SAFETY", "OTHER", "BLOCKLIST",
                            "PROHIBITED_CONTENT", "IMAGE_SAFETY"})
SERVED_KEYS = ("modelVersion",)
REQUEST_ID_KEYS = ("responseId",)
TIMING_KEYS: tuple[str, ...] = ()
TOK_PER_S_BASIS = "wall"
THINKING_TOKEN_KEYS = ("usageMetadata.thoughtsTokenCount",)
CACHE_KEYS = ("usageMetadata.cachedContentTokenCount",)


class Part(TypedDict, total=False):
    """One content part: text, thought or function call."""

    text: str
    thought: Any
    thoughtSignature: str
    inlineData: dict[str, Any]
    fileData: dict[str, Any]
    executableCode: dict[str, Any]
    codeExecutionResult: dict[str, Any]
    functionCall: dict[str, Any]
    functionResponse: dict[str, Any]


class Content(TypedDict, total=False):
    """Content of a candidate."""
    role: str
    parts: list[Part]


class SafetyRating(TypedDict, total=False):
    """Safety rating of a candidate."""
    category: str
    probability: Literal["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"]
    blocked: bool


class Candidate(TypedDict, total=False):
    """One generated candidate."""
    content: Content
    finishReason: Literal["FINISH_REASON_UNSPECIFIED", "STOP", "MAX_TOKENS",
                          "SAFETY", "RECITATION", "FUNCTION_CALL", "OTHER",
                          "BLOCKLIST", "PROHIBITED_CONTENT"]
    safetyRatings: list[SafetyRating]
    index: int


class PromptFeedback(TypedDict, total=False):
    """Feedback when the prompt itself was blocked (candidates is then empty)."""

    blockReason: Literal["BLOCK_REASON_UNSPECIFIED", "SAFETY", "OTHER",
                         "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"]
    safetyRatings: list[SafetyRating]


class UsageMetadata(TypedDict, total=False):
    """Token usage."""
    promptTokenCount: int
    candidatesTokenCount: int
    thoughtsTokenCount: int
    cachedContentTokenCount: int
    toolUsePromptTokenCount: int
    totalTokenCount: int


class GenerateContentResponse(TypedDict, total=False):
    """generateContent response body."""
    candidates: list[Candidate]
    promptFeedback: PromptFeedback
    usageMetadata: UsageMetadata
    modelVersion: str
    responseId: str
