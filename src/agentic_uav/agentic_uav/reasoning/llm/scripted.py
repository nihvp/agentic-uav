"""Deterministic provider for tests: replays a fixed list of responses.

It never inspects the world. Select it explicitly (llm.provider: scripted); it is
not a fallback for a missing API key.
"""

from typing import List

from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolSpec


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: List[LLMResponse]):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        self.calls += 1
        if not self._responses:
            raise RuntimeError("ScriptedProvider ran out of responses.")
        return self._responses.pop(0)
