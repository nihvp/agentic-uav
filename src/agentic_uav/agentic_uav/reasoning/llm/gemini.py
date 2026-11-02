"""Gemini adapter (google-genai SDK) using native function calling."""

import os
import uuid
from typing import List

from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec


class GeminiProvider(LLMProvider):
    def __init__(self, model: str, api_key: str = "", temperature: float = 0.2):
        from google import genai
        from google.genai import types

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self._types = types
        self._client = genai.Client(api_key=key)
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=self._temperature,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=s.name, description=s.description,
                                      parameters_json_schema=s.parameters if s.parameters.get("properties") else None)
                for s in tools
            ])],
        )
        resp = self._client.models.generate_content(
            model=self._model, contents=self._to_contents(messages), config=config)

        calls = [
            ToolCall(id=fc.id or uuid.uuid4().hex[:8], name=fc.name, arguments=dict(fc.args or {}))
            for fc in (resp.function_calls or [])
        ]
        text = "".join(p.text for p in resp.candidates[0].content.parts if getattr(p, "text", None)) \
            if resp.candidates and resp.candidates[0].content else ""
        raw = resp.candidates[0].content if resp.candidates else None
        return LLMResponse(text=text, tool_calls=calls, provider_raw=raw)

    def _to_contents(self, messages: List[Message]):
        t = self._types
        contents = []
        for m in messages:
            if m.role == "user":
                contents.append(t.Content(role="user", parts=[t.Part(text=m.content)]))
            elif m.role == "assistant":
                # Replay the model's own content so thought signatures survive.
                if m.provider_raw is not None:
                    contents.append(m.provider_raw)
                else:
                    parts = [t.Part(text=m.content)] if m.content else []
                    parts += [t.Part(function_call=t.FunctionCall(name=c.name, args=c.arguments)) for c in m.tool_calls]
                    contents.append(t.Content(role="model", parts=parts))
            elif m.role == "tool":
                contents.append(t.Content(role="user", parts=[t.Part(
                    function_response=t.FunctionResponse(name=m.name, response={"result": m.content}))]))
        return contents
