"""Tool registry with decorator-based registration.

Add a tool by writing a function and decorating it with @register_tool. The
decorator inspects a Pydantic args model to generate a JSON schema that the
agent loop injects into the system prompt, and validates model-supplied args
before invoking the function.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_model.model_json_schema(),
        }

    def call(self, raw_args: dict[str, Any]) -> Any:
        try:
            validated = self.args_model(**(raw_args or {}))
        except ValidationError as e:
            raise ToolArgError(str(e)) from e
        return self.func(**validated.model_dump())


class ToolArgError(Exception):
    """Raised when model-supplied args fail validation."""


class ToolNotFoundError(Exception):
    """Raised when the model calls a tool name that isn't registered."""


_REGISTRY: dict[str, Tool] = {}


def register_tool(
    name: str,
    description: str,
    args_model: type[BaseModel],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"Tool already registered: {name}")
        _REGISTRY[name] = Tool(
            name=name, description=description, args_model=args_model, func=func
        )
        return func

    return decorator


def get_tool(name: str) -> Tool:
    if name not in _REGISTRY:
        raise ToolNotFoundError(name)
    return _REGISTRY[name]


def all_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def render_tool_catalog() -> str:
    """Human-readable tool catalog for the system prompt."""
    lines: list[str] = []
    for tool in all_tools():
        schema = tool.args_model.model_json_schema()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        arg_lines = []
        for arg_name, info in props.items():
            type_str = info.get("type", "any")
            req = "required" if arg_name in required else "optional"
            desc = info.get("description", "")
            arg_lines.append(f"    - {arg_name} ({type_str}, {req}): {desc}".rstrip())
        args_block = "\n".join(arg_lines) if arg_lines else "    (no args)"
        lines.append(f"- {tool.name}: {tool.description}\n{args_block}")
    return "\n".join(lines)


def tool_catalog_json() -> str:
    return json.dumps([t.schema() for t in all_tools()], indent=2)
