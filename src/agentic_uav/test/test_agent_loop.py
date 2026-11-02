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
