"""Analyse the captured scenes with each system of the paper's comparison (offline).

Every system reads the same saved frame per scene (captures/scene_NN/rgb_0.png), so
they see identical input. Nothing here reads scenes.json or eval_true_pose.txt: the
scene label and the people are for scoring only.

Systems
  rule     YOLOv11 person detection only, as the paper's rule-based baseline: it reports
           detections and confidence, and has no situation assessment or actions.
  agentic  the same YOLO detections plus the image to a local vision LLM (default the
           LLM_MODEL env, qwen3-vl:8b), which returns the paper's structured JSON:
           person_detected, confidence, situation, surrounding_features,
           recommended_actions, ...

Runs in the agent container (YOLO on the GPU, Ollama on 127.0.0.1:11434):
  python3 -m agentic_uav.experiment.analyse --system rule
  python3 -m agentic_uav.experiment.analyse --system agentic [--model gemma3:4b] [--no-think]
Writes one JSON line per scene to results/<system>[_<model>].jsonl; scenes already in
that file are skipped, so a run can be resumed (--only N ... redoes chosen scenes). Each system first runs once untimed on
a blank frame, so model loading is not counted in the first sample's time.
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ACTIONS = ("NO_INTERVENTION", "CONTINUE_MONITORING", "HOVER_AND_OBSERVE",
           "LAND_AND_DEPLOY_RESCUE_KIT", "ALERT_EMERGENCY_SERVICES")

SYSTEM_PROMPT = f"""You are the reasoning layer of a UAV that monitors a crowd at a mass gathering.
You receive one image from the UAV's camera (about 6 m above the ground, looking ahead and
45 degrees down) and the person detections of the onboard YOLO detector, which can miss
people or be wrong. Look at the image yourself.

Your task: decide whether anyone needs help. An emergency is a person in distress, for
example someone who has collapsed or is lying motionless on the ground. Otherwise the
situation is normal.

Reply with one JSON object and nothing else:
{{"person_detected": true or false,
 "persons_count": number of people you see,
 "confidence": your confidence in this assessment, 0.0 to 1.0,
 "situation": "normal" or "emergency",
 "emergency_description": "who needs help and where in the image; empty if normal",
 "surrounding_features": "what is around: the crowd, the ground, obstacles, access for a landing",
 "recommended_actions": [actions, from: {", ".join(ACTIONS)}]}}"""


# ---- pure parts (unit tested) -------------------------------------------

def capture_frames(root: str) -> List[Tuple[int, str]]:
    """(scene id, frame path) for every captured scene, in scene order."""
    out = []
    for meta in sorted(glob.glob(os.path.join(root, "scene_*", "meta.json"))):
        d = os.path.dirname(meta)
        with open(meta) as f:
            m = json.load(f)
        out.append((int(m["scene_id"]), os.path.join(d, m["frames"][0]["rgb"])))
    return sorted(out)


def user_text(detections: List[Dict[str, Any]]) -> str:
    dets = [{"bbox_xyxy": d["bbox"], "confidence": d["confidence"]} for d in detections]
    return (f"YOLO person detections ({len(dets)}), pixel boxes in a 640x360 image:\n"
            f"{json.dumps(dets)}\nAssess the image and reply with the JSON object.")


def parse_reply(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a model reply (also inside ``` fences or after other text)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _as_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "false", "no"):
        return v.strip().lower() in ("true", "yes")
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _as_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f / 100.0 if 1.0 < f <= 100.0 else f    # "85" -> 0.85


def normalise(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The paper's fields with fixed types; missing or malformed values become None/empty."""
    obj = obj or {}
    actions = obj.get("recommended_actions") or []
    if isinstance(actions, str):
        actions = [actions]
    actions = [str(a).strip() for a in actions if str(a).strip()]
    situation = str(obj.get("situation") or "").strip().lower() or None
    count = obj.get("persons_count")
    try:
        count = int(count) if count is not None else None
    except (TypeError, ValueError):
        count = None
    return {"person_detected": _as_bool(obj.get("person_detected")),
            "persons_count": count,
            "confidence": _as_float(obj.get("confidence")),
            "situation": situation,
            "emergency_description": str(obj.get("emergency_description") or ""),
            "surrounding_features": str(obj.get("surrounding_features") or ""),
            "recommended_actions": actions}


def rule_output(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The rule-based baseline: detections only, no assessment or actions (as in the paper)."""
    confs = [d["confidence"] for d in detections]
    return {"person_detected": bool(detections), "persons_count": len(detections),
            "confidence": max(confs) if confs else 0.0, "situation": None,
            "emergency_description": "", "surrounding_features": "", "recommended_actions": []}


def result_name(system: str, model: str, think: bool = True) -> str:
    if system == "rule":
        return "rule"
    return f"agentic_{re.sub(r'[^A-Za-z0-9.]+', '-', model)}" + ("" if think else "_nothink")


def done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["scene_id"] for line in f if line.strip()}


# ---- systems --------------------------------------------------------------

class Yolo:
    def __init__(self, model: str, confidence: float):
        from agentic_uav.perception.detector import YoloDetector
        self.det = YoloDetector(model=model, min_confidence=confidence)

    def run(self, bgr) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        dets = self.det.detect(bgr)
        return dets, time.perf_counter() - t0


class VisionLLM:
    def __init__(self, model: str, base_url: str, temperature: float, timeout_s: float, think: bool = True):
        from openai import OpenAI
        self.model, self.temperature, self.think = model, temperature, think
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "") or "unused",
                             timeout=timeout_s)

    def run(self, png: bytes, detections) -> Dict[str, Any]:
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        # Ollama: reasoning_effort "none" turns a thinking model's reasoning off.
        extra = {} if self.think else {"reasoning_effort": "none"}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, response_format={"type": "json_object"}, **extra,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": [{"type": "text", "text": user_text(detections)},
                                                   {"type": "image_url", "image_url": {"url": url}}]}])
        elapsed = time.perf_counter() - t0
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return {"time_s": elapsed, "raw": msg.content or "",
                "thinking": getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or "",
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", choices=("rule", "agentic"), required=True)
    ap.add_argument("--captures", default="/experiments/paper/captures")
    ap.add_argument("--results", default="/experiments/paper/results")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3-vl:8b"))
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--yolo-model", default="yolo11s.pt")
    ap.add_argument("--yolo-confidence", type=float, default=0.25)
    ap.add_argument("--no-think", action="store_true",
                    help="turn the model's thinking off (results go to agentic_<model>_nothink.jsonl)")
    ap.add_argument("--only", type=int, nargs="+",
                    help="analyse these scenes again even if done (e.g. after a recapture); "
                         "the new record replaces the old one when scoring")
    a = ap.parse_args()

    import cv2
    import numpy as np

    frames = capture_frames(a.captures)
    if not frames:
        print(f"[analyse] no captures in {a.captures}", file=sys.stderr)
        return 1
    os.makedirs(a.results, exist_ok=True)
    name = result_name(a.system, a.model, think=not a.no_think)
    out_path = os.path.join(a.results, f"{name}.jsonl")
    skip = done_ids(out_path)
    if a.only:
        skip, frames = set(), [(i, p) for i, p in frames if i in set(a.only)]
    todo = [(i, p) for i, p in frames if i not in skip]
    print(f"[analyse] {name}: {len(frames)} captured scenes, {len(skip)} already done, {len(todo)} to do")

    yolo = Yolo(a.yolo_model, a.yolo_confidence)
    llm = (VisionLLM(a.model, a.base_url, a.temperature, a.timeout, think=not a.no_think)
           if a.system == "agentic" else None)
    blank = np.full((360, 640, 3), 180, np.uint8)
    yolo.run(blank)                                     # warm-up, untimed
    if llm:
        warm = llm.run(cv2.imencode(".png", blank)[1].tobytes(), [])
        if a.no_think and warm["thinking"]:
            # qwen3-vl:8b ignores the off switch (a Thinking build); use an Instruct tag.
            print(f"[analyse] --no-think, but {a.model} still thought ({len(warm['thinking'])} chars). "
                  "This model cannot turn thinking off; use an instruct variant "
                  "(e.g. --model qwen3-vl:8b-instruct) instead.", file=sys.stderr)
            return 2

    with open(out_path, "a") as out:
        for scene_id, path in todo:
            bgr = cv2.imread(path)
            if bgr is None:
                print(f"[analyse] scene {scene_id}: cannot read {path}", file=sys.stderr)
                continue
            png = open(path, "rb").read()
            t0 = time.perf_counter()
            dets, yolo_s = yolo.run(bgr)
            rec = {"scene_id": scene_id, "system": a.system, "frame": os.path.relpath(path, a.captures),
                   "yolo": {"model": a.yolo_model, "time_s": round(yolo_s, 4), "detections": dets}}
            if llm:
                r = llm.run(png, dets)
                parsed = parse_reply(r["raw"])
                rec["llm"] = {"model": a.model, "temperature": a.temperature, "think": not a.no_think,
                              "parse_ok": parsed is not None,
                              **{k: (round(v, 3) if k == "time_s" else v) for k, v in r.items()}}
                rec["output"] = normalise(parsed)
            else:
                rec["output"] = rule_output(dets)
            rec["processing_time_s"] = round(time.perf_counter() - t0, 4)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            o = rec["output"]
            print(f"[analyse] scene {scene_id}: {rec['processing_time_s']:.2f} s, YOLO {len(dets)} persons, "
                  f"situation {o['situation']}, actions {o['recommended_actions']}")
    print(f"[analyse] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
