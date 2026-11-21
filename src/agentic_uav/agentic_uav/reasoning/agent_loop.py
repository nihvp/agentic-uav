"""Multi-step ReAct-style loop over an LLMProvider and a ToolRegistry."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic_uav.reasoning.history import approx_tokens, compact_tool_results
from agentic_uav.reasoning.llm.base import LLMProvider, Message
from agentic_uav.reasoning.tools import ToolRegistry, obj

FINISH_TOOL = "finish_mission"
# Warn once the prompt reaches this fraction of the context window.
CONTEXT_WARN_FRACTION = 0.85


@dataclass
class MissionResult:
    success: bool
    summary: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    # Largest prompt of the mission: server-reported and estimated tokens.
    stats: Dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    def __init__(self, provider: LLMProvider, registry: ToolRegistry, system_prompt: str,
                 max_steps: int = 20, on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 context_tokens: int = 0, keep_full_steps: int = 2):
        self.provider = provider
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _e: None)
        # LLM context window in tokens (0 = unknown, no warning).
        self.context_tokens = context_tokens
        # Steps whose tool results stay complete in the history (see history.py).
        self.keep_full_steps = keep_full_steps
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
        # The goal also goes in the system prompt: if the server ever truncates the
        # history, it drops the oldest messages but keeps the system prompt.
        system = f"{self.system_prompt}\nMission goal: {goal}\n"
        messages: List[Message] = [Message(role="user", content=f"Mission goal: {goal}")]
        tool_messages: List[Tuple[int, Message]] = []
        trace: List[Dict[str, Any]] = []
        stats = {"max_prompt_tokens": 0, "max_approx_prompt_tokens": 0, "context_tokens": self.context_tokens}
        nudges = 0

        def result(success: bool, summary: str) -> MissionResult:
            return MissionResult(success, summary, trace, stats)

        for step in range(1, (max_steps or self.max_steps) + 1):
            compact_tool_results(tool_messages, step, self.keep_full_steps)
            tools = self.registry.specs()
            approx = approx_tokens(system, messages, tools)
            self.on_event({"step": step, "phase": "llm", "approx_prompt_tokens": approx})
            try:
                resp = self.provider.complete(system, messages, tools)
            except Exception as ex:  # network, quota, bad model id: end with the trace so far
                return result(False, f"LLM call failed at step {step}: {type(ex).__name__}: {ex}")
            messages.append(Message(role="assistant", content=resp.text,
                                    tool_calls=resp.tool_calls, provider_raw=resp.provider_raw))

            prompt_tokens = resp.usage.get("prompt_tokens")
            stats["max_approx_prompt_tokens"] = max(stats["max_approx_prompt_tokens"], approx)
            if prompt_tokens is not None:
                stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], prompt_tokens)
            size = {"prompt_tokens": prompt_tokens, "approx_prompt_tokens": approx}
            largest = max(prompt_tokens or 0, approx)
            if self.context_tokens and largest >= CONTEXT_WARN_FRACTION * self.context_tokens:
                self.on_event({"step": step, "phase": "context", **size, "context_tokens": self.context_tokens})

            thought = resp.text or (f"[thinking] {resp.thinking}" if resp.thinking else "")
            if not resp.tool_calls:
                nudges += 1
                trace.append({"step": step, "thought": thought, "tool": None, **size})
                if nudges > 2:
                    return result(False, "Model stopped calling tools without finishing.")
                messages.append(Message(role="user", content=f"Call a tool, or {FINISH_TOOL} if done."))
                continue

            for call in resp.tool_calls:
                self.on_event({"step": step, "phase": "start", "thought": thought,
                               "tool": call.name, "arguments": call.arguments, **size})
                res = self.registry.call(call.name, call.arguments)
                entry = {"step": step, "phase": "done", "thought": thought, "tool": call.name,
                         "arguments": call.arguments, "result": res.content, "ok": res.ok, **size}
                trace.append(entry)
                self.on_event(entry)
                msg = Message(role="tool", content=res.to_text(), tool_call_id=call.id, name=call.name)
                messages.append(msg)
                tool_messages.append((step, msg))
                if self._finished is not None:
                    return result(self._finished.success, self._finished.summary)

        return result(False, f"Step budget exhausted ({max_steps or self.max_steps}).")
