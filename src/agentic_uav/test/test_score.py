import json
import os
import random
import sys

import pytest

from agentic_uav.experiment.scenes import EMERGENCY, generate
from agentic_uav.experiment.score import cohens_d, sample_metrics, system_summary


def _rec(scene_id, time_s, output, yolo_n=5, parse_ok=None):
    rec = {"scene_id": scene_id, "processing_time_s": time_s, "output": output,
           "yolo": {"detections": [{"bbox": [0, 0, 1, 1], "confidence": 0.8}] * yolo_n}}
    if parse_ok is not None:
        rec["llm"] = {"parse_ok": parse_ok}
    return rec


def _agentic(scene, rng, recall=1.0):
    emergency = scene.label == EMERGENCY and rng.random() < recall
    return {"person_detected": True, "persons_count": len(scene.people) + rng.choice((-1, 0, 1)),
            "confidence": round(rng.uniform(0.7, 0.95), 2), "situation": "emergency" if emergency else "normal",
            "emergency_description": "", "surrounding_features": "x" * 60,
            "recommended_actions": ["LAND_AND_DEPLOY_RESCUE_KIT"] if emergency else ["CONTINUE_MONITORING"]}


def test_sample_metrics_and_summary():
    scenes = generate()
    e = next(s for s in scenes if s.label == EMERGENCY)
    n = next(s for s in scenes if s.label != EMERGENCY)
    ok = sample_metrics(_rec(e.id, 1.0, {"situation": "emergency", "persons_count": len(e.people),
                                          "recommended_actions": ["ALERT_EMERGENCY_SERVICES"],
                                          "surrounding_features": "short"}), e)
    assert ok["correct_response"] and ok["said_emergency"] and ok["count_error"] == 0 and not ok["car"]
    alarm = sample_metrics(_rec(n.id, 1.0, {"situation": "emergency", "recommended_actions": ["LAND_AND_DEPLOY_RESCUE_KIT"]}), n)
    assert not alarm["correct_response"] and alarm["arr"]
    s = system_summary([ok, alarm])
    assert s["emergency_recall"] == (1, 1) and s["false_alarm_rate"] == (1, 1) and s["situation_accuracy"] == (1, 2)
    rule = system_summary([sample_metrics(_rec(e.id, 0.01, {"person_detected": True, "persons_count": 3,
                                                            "confidence": 0.9, "situation": None,
                                                            "recommended_actions": []}), e)])
    assert rule["emergency_recall"] is None and rule["arr"] == (0, 1)
    failed = sample_metrics(_rec(e.id, 9.0, {"situation": None, "recommended_actions": []}, parse_ok=False), e)
    s2 = system_summary([ok, alarm, failed])
    assert s2["situation_accuracy"] == (1, 3) and s2["emergency_recall"] == (1, 2) and s2["no_assessment"] == 1
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0.0


def test_end_to_end_report(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy")
    from agentic_uav.experiment import score
    from agentic_uav.experiment.scenes import SceneConfig, to_json
    cfg = SceneConfig()
    scenes = generate(cfg)
    (tmp_path / "scenes.json").write_text(to_json(scenes, cfg))
    res = tmp_path / "results"
    res.mkdir()
    rng = random.Random(1)
    with open(res / "rule.jsonl", "w") as f:
        for s in scenes:
            f.write(json.dumps(_rec(s.id, rng.uniform(0.008, 0.012),
                                    {"person_detected": True, "persons_count": 6, "confidence": rng.uniform(0.6, 0.9),
                                     "situation": None, "surrounding_features": "", "recommended_actions": []})) + "\n")
    for name, recall, t in (("agentic_a", 1.0, 1.5), ("agentic_b", 0.5, 0.8)):
        with open(res / f"{name}.jsonl", "w") as f:
            for s in scenes:
                f.write(json.dumps(_rec(s.id, rng.gauss(t, 0.2), _agentic(s, rng, recall), parse_ok=True)) + "\n")
    monkeypatch.setattr(sys, "argv", ["score", "--scenes", str(tmp_path / "scenes.json"), "--results", str(res)])
    assert score.main() == 0
    out = capsys.readouterr().out
    assert "| ARR (any action recommended) | 0/44 (0.0%) | 44/44 (100.0%) | 44/44 (100.0%) |" in out
    summary = json.loads((res / "summary.json").read_text())
    assert summary["systems"]["agentic_a"]["emergency_recall"] == [22, 22]
    assert summary["systems"]["rule"]["situation_accuracy"] is None
    assert "anova_p" in summary["tests"]["processing time"] and os.path.exists(res / "summary.md")


def test_effect_sizes_share_a_sign():
    pytest.importorskip("scipy")
    from agentic_uav.experiment.score import compare_continuous
    pair = compare_continuous({"slow": [5.0, 6.0, 7.0, 6.5], "fast": [1.0, 1.2, 0.9, 1.1]})["pairs"][0]
    assert pair["cohens_d"] > 0 and pair["rank_biserial_r"] > 0
