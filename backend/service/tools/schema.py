"""Builds tool schemas from function signatures."""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from pydantic import create_model

_KEEP = ("type", "description", "enum", "items", "minimum", "maximum")

INTERNAL = {"x-internal": True}


def _flatten(prop: dict) -> dict:
    """Normalize one property to the OpenAI format."""
    if "anyOf" in prop:
        real = next((b for b in prop["anyOf"] if b.get("type") != "null"), {})
        prop = {**real, **{k: v for k, v in prop.items() if k != "anyOf"}}
    return {k: prop[k] for k in _KEEP if k in prop}


def args_model(fn) -> type:
    """Build a Pydantic model from a function signature for schema generation and validation."""
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (hints.get(pname, p.annotation), default)
    return create_model(f"{fn.__name__}_args", __module__=fn.__module__, **fields)


def public_args_model(fn, name: str | None = None) -> type:
    """Return args_model without INTERNAL arguments (for HTTP request bodies)."""
    full = args_model(fn)
    fields = {n: (f.annotation, f) for n, f in full.model_fields.items()
              if f.json_schema_extra != INTERNAL}
    return create_model(name or f"{fn.__name__}_request", __module__=fn.__module__, **fields)


def tool_schema(fn, description: str) -> dict:
    """Build one OpenAI function schema from a tool function and its description."""
    raw = args_model(fn).model_json_schema()
    props = {n: _flatten(p) for n, p in raw.get("properties", {}).items()
             if not p.get("x-internal")}
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(raw.get("required", [])),
            },
        },
    }


def parse_args(fn, raw: dict) -> dict:
    """Validate the arguments sent by the LLM and return the kwargs actually provided."""
    return args_model(fn)(**raw).model_dump(exclude_unset=True)


def call_with(fn, raw: dict, **fixed) -> dict:
    """Call a tool function with the LLM's arguments (declared arguments only)."""
    names = [p for p in inspect.signature(fn).parameters if p not in fixed]
    return fn(**fixed, **{k: raw.get(k) for k in names})


def schema_of(tools: list[dict], name: str) -> dict[str, Any]:
    """Return one schema by name from a schema list."""
    for t in tools:
        if t["function"]["name"] == name:
            return t
    raise KeyError(name)
