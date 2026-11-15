import pytest

genai = pytest.importorskip("google.genai")
from google.genai import errors  # noqa: E402

from agentic_uav.reasoning.llm.gemini import GeminiProvider, retry_delay_s  # noqa: E402

QUOTA_BODY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13s"}]}}


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.sleeps.append(round(dt, 3))
        self.t += dt


def provider(monkeypatch, outcomes, rpm=5.0):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    clock = Clock()
    p = GeminiProvider("m", requests_per_minute=rpm, sleep=clock.sleep, clock=clock)
    calls = []

    def generate_content(**kw):
        calls.append(clock.t)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    monkeypatch.setattr(p._client.models, "generate_content", generate_content)
    return p, clock, calls


def test_retry_delay_parsed_from_quota_error():
    assert retry_delay_s(QUOTA_BODY) == 13.0
    assert retry_delay_s({"error": {}}) == 20.0


def test_requests_are_spaced_to_the_rate_limit(monkeypatch):
    p, clock, calls = provider(monkeypatch, ["a", "b"])
    p._generate([], None)
    clock.t += 2.0                      # the tool call between LLM turns took 2 s
    p._generate([], None)
    assert calls == [0.0, 12.0]         # 60 s / 5 requests


def test_quota_error_waits_server_delay_then_retries(monkeypatch):
    p, clock, calls = provider(monkeypatch, [errors.ClientError(429, QUOTA_BODY), "ok"], rpm=0)
    assert p._generate([], None) == "ok"
    assert clock.sleeps == [14.0] and len(calls) == 2


def test_other_client_errors_are_not_retried(monkeypatch):
    p, _, calls = provider(monkeypatch, [errors.ClientError(404, {"error": {"message": "model not found"}})], rpm=0)
    with pytest.raises(errors.ClientError):
        p._generate([], None)
    assert len(calls) == 1


DAILY_BODY = {"error": {"code": 429, "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41000s"}]}}


def test_long_quota_wait_fails_fast_and_names_the_quota(monkeypatch):
    p, clock, calls = provider(monkeypatch, [errors.ClientError(429, DAILY_BODY)], rpm=0)
    with pytest.raises(RuntimeError, match="PerDay.*41000 s"):
        p._generate([], None)
    assert clock.sleeps == [] and len(calls) == 1
