"""Builds the chat model for each LLM provider."""

from __future__ import annotations

from typing import Optional

import os

from backend.common.config import llm as cfg
from backend.common.config import agent as agent_cfg
from backend.common.config import virtual_hw as vhw_cfg
from backend.repo.llm.providers import PIP_PACKAGE, parse_model

_ANTHROPIC_FALLBACK_MAX_TOKENS = 16000  # Anthropic requires max_tokens

_MAX_RETRIES = 0  # SDK retries are off, transient errors are retried in runtime


def _ollama(name, base_url, num_ctx, timeout_s, max_tokens, think=False):
    """Build a ChatOllama model."""
    from langchain_ollama import ChatOllama

    kw = {"model": name, "base_url": base_url, "num_ctx": num_ctx,
          "client_kwargs": {"timeout": timeout_s}}
    if think:
        kw["reasoning"] = True
    if max_tokens:
        kw["num_predict"] = max_tokens
    return ChatOllama(**kw)


def _openai(name, base_url, timeout_s, max_tokens, effort="", summary=""):
    """Build a ChatOpenAI model."""
    from langchain_openai import ChatOpenAI

    kw = {"model": name, "timeout": timeout_s, "max_retries": _MAX_RETRIES}
    if summary:
        kw["reasoning"] = {"summary": summary, **({"effort": effort} if effort else {})}
        kw["output_version"] = "responses/v1"
    elif effort:
        kw["reasoning_effort"] = effort
    if base_url:
        kw["base_url"] = base_url
    if max_tokens:
        kw["max_tokens"] = max_tokens
    return ChatOpenAI(**kw)


def _anthropic(name, base_url, timeout_s, max_tokens, adaptive=True, effort="", display=""):
    """Build a ChatAnthropic model."""
    from langchain_anthropic import ChatAnthropic

    kw = {"model": name, "timeout": timeout_s, "max_retries": _MAX_RETRIES,
          "max_tokens": max_tokens or _ANTHROPIC_FALLBACK_MAX_TOKENS}
    if adaptive:
        kw["thinking"] = {"type": "adaptive", **({"display": display} if display else {})}
    if effort:
        kw["effort"] = effort
    if base_url:
        kw["anthropic_api_url"] = base_url
    return ChatAnthropic(**kw)


def _google(name, base_url, timeout_s, max_tokens, include_thoughts=False,
            effort="", budget_tokens=None):
    """Build a ChatGoogleGenerativeAI model."""
    import os

    from langchain_google_genai import ChatGoogleGenerativeAI

    kw = {"model": name, "timeout": timeout_s, "max_retries": _MAX_RETRIES}
    if include_thoughts:
        kw["include_thoughts"] = True
    if effort:
        kw["reasoning_effort"] = effort
    if budget_tokens is not None:
        kw["thinking_budget"] = budget_tokens
    if max_tokens:
        kw["max_output_tokens"] = max_tokens
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        kw["google_api_key"] = key
    return ChatGoogleGenerativeAI(**kw)


def build(spec: str, *, base_url: str = "", num_ctx: int = 0, timeout_s: float = 600.0,
          max_tokens: int = 0, tools: Optional[list] = None):
    """Build a LangChain chat model from a model spec string, applying the thinking settings from model_profiles.json."""
    from backend.repo.llm import model_profiles

    provider, name = parse_model(spec)
    prof = model_profiles.resolve(f"{provider}:{name}")
    mode = prof.get("mode", "")
    try:
        if provider == "ollama":
            llm = _ollama(name, base_url, num_ctx, timeout_s, max_tokens,
                          think=(mode == "on"))
        elif provider == "openai":
            llm = _openai(name, base_url, timeout_s, max_tokens,
                          effort=prof.get("effort", "") if mode == "effort" else "",
                          summary=prof.get("summary", ""))
        elif provider == "anthropic":
            llm = _anthropic(name, base_url, timeout_s, max_tokens,
                             adaptive=(mode == "adaptive"),
                             effort=prof.get("effort", ""),
                             display=prof.get("display", ""))
        elif provider == "google":
            llm = _google(name, base_url, timeout_s, max_tokens,
                          include_thoughts=bool(prof.get("include_thoughts")),
                          effort=prof.get("effort", ""),
                          budget_tokens=(prof.get("budget_tokens") if mode == "budget" else None))
        else:
            raise ValueError(f"Unknown provider {provider!r}")
    except ImportError as e:
        raise ImportError(
            f"the {provider} provider requires {PIP_PACKAGE[provider]} "
            f"(pip install {PIP_PACKAGE[provider]}). Cause: {e}") from e
    except TypeError as e:
        raise TypeError(
            f"the langchain package rejected the constructor arguments for {provider}:{name} - "
            f"adjust the thinking block in backend/repo/llm/model_profiles.json to that package version. "
            f"Cause: {e}") from e
    if not tools:
        return llm
    if provider == WEB_SEARCH_FUNCTION_PROVIDER:
        return llm.bind_tools(tools)
    native = NATIVE_WEB_SEARCH[provider]
    tools = [native if _fn_name(t) == "web_search" else t for t in tools]
    if provider == "google":
        import copy

        from langchain_google_genai._function_utils import convert_to_genai_function_declarations
        return llm.bind(tools=convert_to_genai_function_declarations(copy.deepcopy(tools)),
                        tool_config={"include_server_side_tool_invocations": True})
    if provider == "openai":
        llm = llm.model_copy(update={"include": ["web_search_call.action.sources"]})
    return llm.bind_tools(tools)


NATIVE_WEB_SEARCH = {
    "anthropic": {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
    "openai": {"type": "web_search"},
    "google": {"google_search": {}},
}

WEB_SEARCH_FUNCTION_PROVIDER = "ollama"


def _fn_name(tool) -> str:
    """Return the function name from a tool schema, or ""."""
    return ((tool or {}).get("function") or {}).get("name", "") if isinstance(tool, dict) else ""


def describe() -> str:
    """Return a one-line summary of the LLM configuration for logs and health responses."""
    hw = f"virtual:{vhw_cfg.scene_label()}" if vhw_cfg.VIRTUAL_HW else "real-hw"

    from backend.repo.llm import providers
    model = providers.describe(cfg.OLLAMA_MODEL)
    provider, _ = providers.parse_model(cfg.OLLAMA_MODEL)

    if provider == "ollama":
        where = f" @ {cfg.OLLAMA_HOST}"
        knobs = f"num_ctx={cfg.NUM_CTX}, "
    else:
        where = f" @ {cfg.API_BASE_URL}" if cfg.API_BASE_URL else ""
        knobs = ""
    if cfg.MAX_TOKENS:
        cap = f"max_tokens={cfg.MAX_TOKENS}"
    elif provider == "anthropic":
        cap = f"max_tokens={_ANTHROPIC_FALLBACK_MAX_TOKENS} (required by anthropic · fallback value)"
    else:
        cap = "max_tokens=provider-default"

    key_env = cfg.API_KEY_ENV.get(provider) or ""
    warn = f"  ⚠ {key_env} not set" if key_env and not os.getenv(key_env, "").strip() else ""

    ollama_fn = provider == WEB_SEARCH_FUNCTION_PROVIDER
    search = "ollama-api" if ollama_fn else f"{provider}-native"
    if ollama_fn and not os.getenv("OLLAMA_API_KEY", "").strip():
        warn += "  ⚠ OLLAMA_API_KEY not set (web_search unavailable)"

    return (f"{agent_cfg.AGENT_ARCH} · {model}{where} "
            f"({knobs}{cap}, timeout={cfg.LLM_TIMEOUT_S:.0f}s, "
            f"chat_isolated={agent_cfg.CHAT_SESSION_ISOLATED}, search={search}, {hw}){warn}")


__all__ = ["build", "describe"]
