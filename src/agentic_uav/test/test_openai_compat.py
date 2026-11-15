from types import SimpleNamespace as NS

from agentic_uav.reasoning.llm.base import Message, ToolCall, ToolSpec
from agentic_uav.reasoning.llm.openai_compat import OpenAICompatProvider, parse_arguments, strip_thinking


class FakeClient:
    """Stands in for openai.OpenAI: records requests, returns queued responses."""

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def reply(content=None, calls=()):
    tool_calls = [NS(id=i, function=NS(name=n, arguments=a)) for i, n, a in calls] or None
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


TOOLS = [ToolSpec("navigate_to", "Fly to a point.",
                  {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}),
         ToolSpec("scan", "Look around.", {"type": "object", "properties": {}, "required": []})]


def test_tool_call_is_parsed():
    client = FakeClient([reply("<think>plan it</think>Flying there.", [("c1", "navigate_to", '{"x": 8.0}')])])
    resp = OpenAICompatProvider("m", client=client).complete("sys", [Message("user", "go")], TOOLS)
    assert resp.text == "Flying there."
    assert resp.tool_calls == [ToolCall("c1", "navigate_to", {"x": 8.0})]


def test_request_carries_system_tools_and_history():
    client = FakeClient([reply("done")])
    history = [
        Message("user", "Mission goal: find"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "scan", {})]),
        Message("tool", '{"ok": true}', tool_call_id="c1", name="scan"),
    ]
    OpenAICompatProvider("qwen", client=client, temperature=0.3).complete("sys", history, TOOLS)
    req = client.requests[0]
    assert req["model"] == "qwen" and req["temperature"] == 0.3
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "tool"]
    assert req["messages"][2]["tool_calls"][0]["function"] == {"name": "scan", "arguments": "{}"}
    assert req["messages"][2]["content"] is None
    assert req["messages"][3]["tool_call_id"] == "c1"
    assert [t["function"]["name"] for t in req["tools"]] == ["navigate_to", "scan"]
    assert "reasoning_effort" not in req


def test_bad_arguments_reach_the_registry_as_an_error():
    assert parse_arguments('{"x": 1') == {"invalid_json_arguments": '{"x": 1'}
    assert parse_arguments("") == {}
    assert parse_arguments({"x": 1}) == {"x": 1}


def test_unclosed_think_is_dropped():
    assert strip_thinking("<think>still going") == ""
    assert strip_thinking("plain") == "plain"
