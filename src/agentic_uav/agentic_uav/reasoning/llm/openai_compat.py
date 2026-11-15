"""Adapter for OpenAI-compatible chat servers with tool calling.

Covers the local LLM service (Ollama, default http://127.0.0.1:11434/v1) and also
vLLM, LM Studio or any hosted endpoint that speaks /v1/chat/completions with `tools`.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec

# Reasoning models (e.g. Qwen3) may put their chain of thought in the content.
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


class OpenAICompatProvider(LLMProvider):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434/v1", api_key: str = "",
                 temperature: Optional[float] = None, timeout_s: float = 120.0, max_retries: int = 2,
                 reasoning_effort: str = "", client: Any = None):
        if client is None:
            from openai import OpenAI
            # Local servers ignore the key, but the SDK requires one.
            client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("LLM_API_KEY", "") or "unused",
                            timeout=timeout_s, max_retries=max_retries)
        self._client = client
        self._model = model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def complete(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        kwargs: Dict[str, Any] = {"model": self._model, "messages": to_openai_messages(system, messages)}
        if tools:
            kwargs["tools"] = [to_openai_tool(s) for s in tools]
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        resp = self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise RuntimeError("LLM server returned no choices.")
        msg = resp.choices[0].message
        calls = [ToolCall(id=c.id or uuid.uuid4().hex[:8], name=c.function.name,
                          arguments=parse_arguments(c.function.arguments))
                 for c in (msg.tool_calls or [])]
        return LLMResponse(text=strip_thinking(msg.content or ""), tool_calls=calls)


def to_openai_tool(spec: ToolSpec) -> Dict[str, Any]:
    params = spec.parameters if spec.parameters.get("properties") else {"type": "object", "properties": {}}
    return {"type": "function", "function": {"name": spec.name, "description": spec.description,
                                             "parameters": params}}


def to_openai_messages(system: str, messages: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.role == "assistant":
            entry: Dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                       for c in m.tool_calls]
            out.append(entry)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def parse_arguments(raw: Any) -> Dict[str, Any]:
    """Tool-call arguments arrive as a JSON string (some servers send a dict)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        # Passed through so the registry rejects it and the model sees why.
        return {"invalid_json_arguments": raw}
    return args if isinstance(args, dict) else {"invalid_json_arguments": raw}


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text).strip()
