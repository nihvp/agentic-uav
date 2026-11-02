"""Multi-step ReAct-style loop over an LLMProvider and a ToolRegistry."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentic_uav.reasoning.llm.base import LLMProvider, Message
from agentic_uav.reasoning.tools import ToolRegistry, obj

FINISH_TOOL = "finish_mission"


@dataclass
class MissionResult:
    success: bool
    summary: str
    trace: List[Dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    def __init__(self, provider: LLMProvider, registry: ToolRegistry, system_prompt: str,
                 max_steps: int = 20, on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.provider = provider
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _e: None)
        self._finished: Optional[MissionResult] = None
        if FINISH_TOOL not in {s.name for s in registry.specs()}:
            registry.register(
                FINISH_TOOL,
                "End the mission. success=false if the goal could not be achieved.",
                obj({"success": {"type": "boolean"}, "summary": {"type": "string"}}, ["success", "summary"]),
                self._finish,
            )

    def _finish(self, success: bool, summary: str) -> str:
        self._finished = MissionResult(bool(success), summary)
        return "Mission finished."

    def run(self, goal: str, max_steps: Optional[int] = None) -> MissionResult:
        self._finished = None
        messages: List[Message] = [Message(role="user", content=f"Mission goal: {goal}")]
        trace: List[Dict[str, Any]] = []
        nudges = 0

        for step in range(1, (max_steps or self.max_steps) + 1):
            resp = self.provider.complete(self.system_prompt, messages, self.registry.specs())
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=resp.tool_calls, provider_raw=resp.provider_raw))

            if not resp.tool_calls:
                nudges += 1
                trace.append({"step": step, "thought": resp.text, "tool": None})
                if nudges > 2:
                    return MissionResult(False, "Model stopped calling tools without finishing.", trace)
                messages.append(Message(role="user", content=f"Call a tool, or {FINISH_TOOL} if done."))
                continue

            for call in resp.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                entry = {"step": step, "thought": resp.text, "tool": call.name,
                         "arguments": call.arguments, "result": result.content, "ok": result.ok}
                trace.append(entry)
                self.on_event(entry)
                messages.append(Message(role="tool", content=result.to_text(),
                                        tool_call_id=call.id, name=call.name))
                if self._finished is not None:
                    self._finished.trace = trace
                    return self._finished

        return MissionResult(False, f"Step budget exhausted ({max_steps or self.max_steps}).", trace)
