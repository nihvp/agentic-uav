"""Provider-agnostic LLM interface used by the agent loop."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # Provider-specific payload to replay verbatim (e.g. Gemini thought signatures).
    provider_raw: Any = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall]
    provider_raw: Any = None


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        """Return the model's next turn: free text and/or tool calls."""
