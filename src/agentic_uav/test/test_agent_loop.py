from agentic_uav.reasoning.agent_loop import AgentLoop
from agentic_uav.reasoning.llm.base import LLMResponse, ToolCall
from agentic_uav.reasoning.llm.scripted import ScriptedProvider
from agentic_uav.reasoning.tools import ToolRegistry, obj


def call(name, **args):
    return LLMResponse(text="", tool_calls=[ToolCall(id=name, name=name, arguments=args)])


def registry(log):
    r = ToolRegistry()
    r.register("takeoff", "", obj({"altitude": {"type": "number"}}, ["altitude"]),
               lambda altitude: log.append(("takeoff", altitude)) or "ok")

    def fail():
        raise RuntimeError("navigation rejected")
    r.register("navigate_to", "", obj({}), fail)
    return r


def test_runs_tools_and_finishes():
    log = []
    provider = ScriptedProvider([call("takeoff", altitude=5), call("finish_mission", success=True, summary="done")])
    result = AgentLoop(provider, registry(log), "sys").run("go")
    assert result.success and result.summary == "done"
    assert log == [("takeoff", 5)]
    assert [t["tool"] for t in result.trace] == ["takeoff", "finish_mission"]


def test_tool_errors_become_observations():
    provider = ScriptedProvider([call("navigate_to"), call("finish_mission", success=False, summary="blocked")])
    result = AgentLoop(provider, registry([]), "sys").run("go")
    assert result.trace[0]["ok"] is False and "rejected" in result.trace[0]["result"]
    assert not result.success


def test_missing_args_and_budget():
    provider = ScriptedProvider([call("takeoff")] * 3)
    result = AgentLoop(provider, registry([]), "sys", max_steps=3).run("go")
    assert not result.success and "budget" in result.summary
    assert "Missing required" in result.trace[0]["result"]


class Recording(ScriptedProvider):
    """Keeps a copy of every request (system prompt, message contents)."""

    def __init__(self, responses):
        super().__init__(responses)
        self.requests = []

    def complete(self, system, messages, tools):
        self.requests.append((system, [(m.role, m.content) for m in messages]))
        return super().complete(system, messages, tools)


def test_goal_is_in_the_system_prompt():
    provider = Recording([call("finish_mission", success=True, summary="done")])
    AgentLoop(provider, registry([]), "sys").run("find the survivor")
    assert provider.requests[0][0].startswith("sys") and "Mission goal: find the survivor" in provider.requests[0][0]


def test_thinking_goes_to_the_trace_not_the_history():
    first = LLMResponse(text="", tool_calls=[ToolCall("t", "takeoff", {"altitude": 5})], thinking="climb first")
    provider = Recording([first, call("finish_mission", success=True, summary="done")])
    result = AgentLoop(provider, registry([]), "sys").run("go")
    assert result.trace[0]["thought"] == "[thinking] climb first"
    assert ("assistant", "") in provider.requests[1][1]
    assert not any("climb first" in content for _, content in provider.requests[1][1])


def test_old_tool_results_are_compacted_in_the_history_but_not_the_trace():
    r = ToolRegistry()
    r.register("scan", "", obj({}), lambda: {"objects": [{"id": 1, "label": "survivor"}], "known_objects": []})
    provider = Recording([call("scan")] * 4 + [call("finish_mission", success=True, summary="done")])
    result = AgentLoop(provider, r, "sys", keep_full_steps=2).run("go")
    last_history = provider.requests[-1][1]
    tool_contents = [c for role, c in last_history if role == "tool"]
    assert "(omitted" in tool_contents[0] and "(omitted" in tool_contents[1]
    assert "(omitted" not in tool_contents[2] and "(omitted" not in tool_contents[3]
    assert '"label": "survivor"' in tool_contents[3]
    assert all(t["result"]["objects"] == [{"id": 1, "label": "survivor"}] for t in result.trace[:4])


def test_usage_stats_and_context_warning():
    def with_usage(resp, tokens):
        resp.usage = {"prompt_tokens": tokens}
        return resp

    provider = ScriptedProvider([with_usage(call("takeoff", altitude=5), 500),
                                 with_usage(call("finish_mission", success=True, summary="done"), 950)])
    events = []
    result = AgentLoop(provider, registry([]), "sys", context_tokens=1000, on_event=events.append).run("go")
    assert result.stats["max_prompt_tokens"] == 950 and result.stats["context_tokens"] == 1000
    assert result.stats["max_approx_prompt_tokens"] > 0
    assert [e["step"] for e in events if e["phase"] == "context"] == [2]
    assert result.trace[0]["prompt_tokens"] == 500


def test_llm_failure_ends_mission_with_trace():
    class Flaky(ScriptedProvider):
        def complete(self, system, messages, tools):
            if len(messages) > 2:
                raise TimeoutError("read timed out")
            return super().complete(system, messages, tools)

    events = []
    result = AgentLoop(Flaky([call("takeoff", altitude=5)]), registry([]), "sys",
                       on_event=events.append).run("go")
    assert not result.success and "TimeoutError" in result.summary
    assert [t["tool"] for t in result.trace] == ["takeoff"]
    assert [e["phase"] for e in events] == ["llm", "start", "done", "llm"]
