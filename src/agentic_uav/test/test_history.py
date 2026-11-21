import json

from agentic_uav.reasoning.history import (CUMULATIVE_NOTE, SNAPSHOT_NOTE, approx_tokens,
                                           compact_tool_results)
from agentic_uav.reasoning.llm.base import Message, ToolCall, ToolSpec
from agentic_uav.reasoning.tools import ToolResult


def tool_msg(result, ok=True):
    return Message("tool", ToolResult(ok, result).to_text(), tool_call_id="c", name="t")


def result_of(msg):
    return json.loads(msg.content)["result"]


def nav(step):
    return {"message": "Reached target.", "final_position": [step, 0, 5],
            "scene": [{"label": "survivor"}], "above": [], "known_objects": [{"ref": f"survivor-{step}"}]}


def test_old_snapshots_and_superseded_known_objects_are_replaced():
    entries = [(s, tool_msg(nav(s))) for s in (1, 2, 3, 4)]
    compact_tool_results(entries, current_step=5, keep_steps=2)
    r = [result_of(m) for _, m in entries]
    # Steps 1-2 are older than the last two: scene dropped. Steps 3-4 keep it.
    assert [x["scene"] == SNAPSHOT_NOTE for x in r] == [True, True, False, False]
    # Only the newest known_objects survives, whatever its age.
    assert [x["known_objects"] for x in r[:3]] == [CUMULATIVE_NOTE] * 3
    assert r[3]["known_objects"] == [{"ref": "survivor-4"}]
    # Everything else is untouched.
    assert [x["final_position"] for x in r] == [[1, 0, 5], [2, 0, 5], [3, 0, 5], [4, 0, 5]]


def test_newest_known_objects_kept_when_later_tools_lack_it():
    entries = [(1, tool_msg(nav(1))), (2, tool_msg({"armed": True})), (3, tool_msg("Mission finished."))]
    compact_tool_results(entries, current_step=6, keep_steps=2)
    assert result_of(entries[0][1])["known_objects"] == [{"ref": "survivor-1"}]
    assert result_of(entries[0][1])["scene"] == SNAPSHOT_NOTE


def test_scan_objects_compacted_and_non_dict_results_left_alone():
    scan = {"position": [0, 0, 5], "objects": [{"id": 1}], "known_objects": []}
    entries = [(1, tool_msg(scan)), (1, tool_msg("RuntimeError: rejected", ok=False)),
               (2, Message("tool", "not json", tool_call_id="c", name="t"))]
    compact_tool_results(entries, current_step=4, keep_steps=2)
    assert result_of(entries[0][1])["objects"] == SNAPSHOT_NOTE
    assert result_of(entries[1][1]) == "RuntimeError: rejected"
    assert entries[2][1].content == "not json"


def test_compaction_is_idempotent():
    entries = [(s, tool_msg(nav(s))) for s in (1, 2, 3)]
    compact_tool_results(entries, current_step=6)
    first = [m.content for _, m in entries]
    compact_tool_results(entries, current_step=7)
    assert [m.content for _, m in entries] == first


def test_approx_tokens_grows_with_history():
    tools = [ToolSpec("scan", "Look around.", {"type": "object", "properties": {}})]
    short = approx_tokens("sys", [Message("user", "go")], tools)
    longer = approx_tokens("sys", [Message("user", "go"),
                                   Message("assistant", "", tool_calls=[ToolCall("1", "scan", {"step_deg": 60})]),
                                   tool_msg(nav(1))], tools)
    assert 0 < short < longer
