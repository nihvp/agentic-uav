"""Tool registry: the single source of truth for tool schemas and handlers."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agentic_uav.reasoning.llm.base import ToolSpec

Handler = Callable[..., Any]


@dataclass
class ToolResult:
    ok: bool
    content: Any

    def to_text(self) -> str:
        return json.dumps({"ok": self.ok, "result": self.content}, default=str)


class ToolRegistry:
    def __init__(self):
        self._specs: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, Handler] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any], handler: Handler):
        if name in self._specs:
            raise ValueError(f"Tool '{name}' already registered.")
        self._specs[name] = ToolSpec(name, description, parameters)
        self._handlers[name] = handler

    def specs(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(False, f"Unknown tool '{name}'. Available: {sorted(self._specs)}")
        missing = [k for k in spec.parameters.get("required", []) if k not in arguments]
        if missing:
            return ToolResult(False, f"Missing required arguments: {missing}")
        try:
            return ToolResult(True, self._handlers[name](**arguments))
        except Exception as ex:  # surfaced to the LLM as an observation
            return ToolResult(False, f"{type(ex).__name__}: {ex}")


def obj(properties: Dict[str, Any], required: List[str] = ()) -> Dict[str, Any]:
    """Shorthand for a JSON Schema object."""
    return {"type": "object", "properties": properties, "required": list(required)}
