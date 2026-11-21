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
    # The model's reasoning, if the server returns it separately. Kept for the trace,
    # not replayed into the history (it would only fill the context window).
    thinking: str = ""
    # Token counts the server reported: prompt_tokens, completion_tokens (empty if none).
    usage: Dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        """Return the model's next turn: free text and/or tool calls."""
