import json

from agentic_uav.experiment.analyse import (ACTIONS, SYSTEM_PROMPT, capture_frames, done_ids, normalise,
                                            parse_reply, result_name, rule_output, user_text)


def test_parse_reply_variants():
    obj = {"situation": "emergency", "recommended_actions": ["LAND_AND_DEPLOY_RESCUE_KIT"]}
    assert parse_reply(json.dumps(obj)) == obj
    assert parse_reply("```json\n" + json.dumps(obj) + "\n```") == obj
    assert parse_reply("<think>hmm {not json}</think>Here: " + json.dumps(obj)) == obj
    assert parse_reply("no json here") is None and parse_reply("") is None


def test_normalise_types():
    n = normalise({"person_detected": "yes", "persons_count": "12", "confidence": 85,
                   "situation": " Emergency ", "recommended_actions": "ALERT_EMERGENCY_SERVICES"})
    assert n["person_detected"] is True and n["persons_count"] == 12 and n["confidence"] == 0.85
    assert n["situation"] == "emergency" and n["recommended_actions"] == ["ALERT_EMERGENCY_SERVICES"]
    empty = normalise(None)
    assert empty["person_detected"] is None and empty["recommended_actions"] == [] and empty["situation"] is None
    assert normalise({"confidence": 1.0})["confidence"] == 1.0


def test_rule_baseline_has_no_assessment():
    o = rule_output([{"bbox": [0, 0, 1, 1], "confidence": 0.7}, {"bbox": [0, 0, 1, 1], "confidence": 0.9}])
    assert o["person_detected"] and o["persons_count"] == 2 and o["confidence"] == 0.9
    assert o["recommended_actions"] == [] and o["surrounding_features"] == "" and o["situation"] is None
    assert rule_output([])["person_detected"] is False


def test_prompt_lists_actions_and_has_no_ground_truth():
    assert all(a in SYSTEM_PROMPT for a in ACTIONS)
    text = user_text([{"bbox": [1, 2, 3, 4], "confidence": 0.5, "label": "person", "source": "yolo"}])
    assert "[1, 2, 3, 4]" in text and "source" not in text


def test_capture_frames_and_resume(tmp_path):
    for sid in (2, 1):
        d = tmp_path / f"scene_{sid:02d}"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"scene_id": sid, "frames": [{"rgb": "rgb_0.png"}]}))
    frames = capture_frames(str(tmp_path))
    assert [i for i, _ in frames] == [1, 2] and frames[0][1].endswith("scene_01/rgb_0.png")
    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps({"scene_id": 1}) + "\n")
    assert done_ids(str(res)) == {1} and done_ids(str(tmp_path / "none.jsonl")) == set()
    assert result_name("rule", "x") == "rule" and result_name("agentic", "qwen3-vl:8b") == "agentic_qwen3-vl-8b"
