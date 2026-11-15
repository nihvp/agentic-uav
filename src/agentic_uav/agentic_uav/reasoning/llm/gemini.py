"""Gemini adapter (google-genai SDK) using native function calling."""

import os
import re
import time
import uuid
from typing import List, Optional

from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec


class GeminiProvider(LLMProvider):
    # temperature=None keeps the model default; Google advises the default (1.0) for
    # Gemini 3 models, where lower values can make tool use loop.
    def __init__(self, model: str, api_key: str = "", temperature: Optional[float] = None,
                 timeout_s: float = 90.0, attempts: int = 3, requests_per_minute: float = 0.0,
                 rate_limit_retries: int = 4, max_retry_wait_s: float = 60.0, log=None,
                 thinking_level: str = "",
                 sleep=time.sleep, clock=time.monotonic):
        from google import genai
        from google.genai import types

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self._types = types
        # Without a timeout a stalled request blocks the mission forever. The SDK also sends
        # it as the server-side deadline, so a 504 means the model ran out of time: retrying
        # the same request would too, so only 500/502/503 are retried. 429 (quota) is
        # handled below with the server's delay.
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=attempts, http_status_codes=[500, 502, 503])))
        # Gemini 3 models think at "high" by default, which can take minutes per step with
        # many tools; picking the next flight action needs little. "" = model default.
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level.upper()) if thinking_level else None
        from google.genai import errors
        self._api_error = errors.APIError
        # Free-tier keys allow a few requests per minute; space calls out instead of failing.
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._rate_limit_retries = rate_limit_retries
        # A per-minute limit asks for seconds; a daily limit asks for hours. Fail on those.
        self._max_retry_wait = max_retry_wait_s
        self._log = log or (lambda _msg: None)
        self._sleep, self._clock = sleep, clock
        self._last_request = None
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=self._temperature,
            thinking_config=self._thinking,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=s.name, description=s.description,
                                      parameters_json_schema=s.parameters if s.parameters.get("properties") else None)
                for s in tools
            ])],
        )
        resp = self._generate(self._to_contents(messages), config)

        calls = [
            ToolCall(id=fc.id or uuid.uuid4().hex[:8], name=fc.name, arguments=dict(fc.args or {}))
            for fc in (resp.function_calls or [])
        ]
        text = "".join(p.text for p in resp.candidates[0].content.parts if getattr(p, "text", None)) \
            if resp.candidates and resp.candidates[0].content else ""
        raw = resp.candidates[0].content if resp.candidates else None
        return LLMResponse(text=text, tool_calls=calls, provider_raw=raw)

    def _generate(self, contents, config):
        for attempt in range(self._rate_limit_retries + 1):
            if self._last_request is not None and self._min_interval:
                wait = self._min_interval - (self._clock() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            try:
                return self._client.models.generate_content(model=self._model, contents=contents, config=config)
            except self._api_error as ex:
                if ex.code != 429:
                    raise
                delay = retry_delay_s(ex.details)
                quota = quota_id(ex.details)
                if attempt == self._rate_limit_retries or delay > self._max_retry_wait:
                    raise RuntimeError(f"Gemini quota exhausted ({quota}); server asks to wait {delay:.0f} s. "
                                       f"Check usage at https://ai.dev/rate-limit") from ex
                self._log(f"Gemini quota hit ({quota}); waiting {delay + 1.0:.0f} s, "
                          f"retry {attempt + 1}/{self._rate_limit_retries}.")
                self._sleep(delay + 1.0)

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


def retry_delay_s(details, default: float = 20.0) -> float:
    """Server-suggested wait from a 429 body (RetryInfo.retryDelay, e.g. "13s")."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(details))
    return float(match.group(1)) if match else default


def quota_id(details) -> str:
    """Which quota a 429 hit, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier."""
    match = re.search(r"quotaId['\"]?\s*:\s*['\"]([^'\"]+)", str(details))
    return match.group(1) if match else "unknown quota"
