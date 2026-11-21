"""Keep the mission history inside the LLM's context window.

Every navigate_to / scan / takeoff result carries the scene on arrival and the
mission's known_objects, so a 40-step mission can outgrow a 16k context. Ollama then
drops the oldest messages without an error, which can remove the mission goal. Two
measures keep the history small without hiding information the model still needs:

* known_objects is cumulative, so only the newest copy is kept; older ones are
  replaced by a note.
* scene and scan objects are snapshots; they are kept in full for the last few
  steps and replaced by a note in older results.

Only the copy sent to the LLM is compacted. The mission trace keeps every result.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentic_uav.reasoning.llm.base import Message, ToolSpec

SNAPSHOT_KEYS = ("scene", "objects")
CUMULATIVE_KEY = "known_objects"
SNAPSHOT_NOTE = "(omitted from older results to save context)"
CUMULATIVE_NOTE = "(superseded by a later known_objects)"


def _result_dict(msg: Message) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(envelope, result) when a tool message holds a JSON object result, else None."""
    try:
        data = json.loads(msg.content)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        return data, data["result"]
    return None


def compact_tool_results(entries: Sequence[Tuple[int, Message]], current_step: int, keep_steps: int = 2):
    """Shrink older tool results in place. entries = [(step, tool message)], oldest first.

    Results from the last `keep_steps` steps before `current_step` keep their snapshots.
    """
    parsed = [(step, msg, _result_dict(msg)) for step, msg in entries]
    newest_known = max((i for i, (_, _, p) in enumerate(parsed)
                        if p and isinstance(p[1].get(CUMULATIVE_KEY), list)), default=None)
    for i, (step, msg, p) in enumerate(parsed):
        if p is None:
            continue
        envelope, result = p
        changed = False
        if i != newest_known and isinstance(result.get(CUMULATIVE_KEY), list):
            result[CUMULATIVE_KEY] = CUMULATIVE_NOTE
            changed = True
        if step < current_step - keep_steps:
            for key in SNAPSHOT_KEYS:
                if key in result and not isinstance(result[key], str):
                    result[key] = SNAPSHOT_NOTE
                    changed = True
        if changed:
            msg.content = json.dumps(envelope, default=str)


def approx_tokens(system: str, messages: List[Message], tools: List[ToolSpec]) -> int:
    """Rough prompt size (~3.5 characters per token for this JSON-heavy text).

    A cross-check on the server's count, which cannot show what it truncated away.
    """
    chars = len(system)
    for m in messages:
        chars += len(m.content or "")
        chars += sum(len(c.name) + len(json.dumps(c.arguments, default=str)) for c in m.tool_calls)
    chars += sum(len(s.name) + len(s.description) + len(json.dumps(s.parameters)) for s in tools)
    return int(chars / 3.5)
