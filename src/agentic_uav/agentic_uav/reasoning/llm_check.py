"""One-shot checks that the configured LLM server works for this stack.

Run inside the agent container (no ROS or simulator involved):
  python3 -m agentic_uav.reasoning.llm_check [--provider ollama] [--model qwen3-vl:8b] [--vision]

The tool check sends one request with one tool and expects takeoff(altitude=5). With
--vision it also sends a generated image (a filled circle in a random colour) and
expects the model to name the colour, so it cannot pass by guessing the same answer.
"""

import argparse
import base64
import os
import random
import sys
import time

from agentic_uav.reasoning.llm import Message, ToolSpec, make_provider

COLOURS = {"red": (0, 0, 255), "green": (0, 170, 0), "blue": (255, 0, 0)}   # BGR


def check_tools(a) -> bool:
    tools = [ToolSpec("takeoff", "Take off and hover at the given altitude.",
                      {"type": "object", "properties": {"altitude": {"type": "number", "description": "metres"}},
                       "required": ["altitude"]})]
    provider = make_provider(a.provider, model=a.model, base_url=a.base_url, timeout_s=a.timeout)
    t0 = time.monotonic()
    resp = provider.complete("You control a drone through tools. Act by calling a tool.",
                             [Message("user", "Take off to 5 metres.")], tools)
    print(f"[tools] model={a.model} took {time.monotonic() - t0:.1f} s")
    print(f"[tools] text: {resp.text!r}")
    print(f"[tools] tool calls: {[(c.name, c.arguments) for c in resp.tool_calls]}")
    ok = any(c.name == "takeoff" and abs(float(c.arguments.get("altitude", 0)) - 5.0) < 0.01
             for c in resp.tool_calls)
    print("[tools] PASS" if ok else "[tools] FAIL: expected takeoff(altitude=5)")
    return ok


def test_image(colour: str) -> bytes:
    import cv2
    import numpy as np
    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.circle(img, (160, 120), 70, COLOURS[colour], -1)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Could not encode the test image.")
    return png.tobytes()


def check_vision(a) -> bool:
    from openai import OpenAI
    colour = random.choice(sorted(COLOURS))
    url = "data:image/png;base64," + base64.b64encode(test_image(colour)).decode()
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get("LLM_API_KEY", "") or "unused", timeout=a.timeout)
    t0 = time.monotonic()
    resp = client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": [
        {"type": "text", "text": "What colour is the filled circle? Answer with one word: red, green or blue."},
        {"type": "image_url", "image_url": {"url": url}}]}])
    answer = (resp.choices[0].message.content or "").strip()
    print(f"[vision] model={a.model} took {time.monotonic() - t0:.1f} s; circle was {colour}; answer: {answer!r}")
    others = [c for c in COLOURS if c != colour]
    ok = colour in answer.lower() and not any(c in answer.lower() for c in others)
    print("[vision] PASS" if ok else f"[vision] FAIL: expected '{colour}' (a text-only model cannot see the image)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3-vl:8b"))
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--timeout", type=float, default=300.0, help="first call also loads the model into VRAM")
    ap.add_argument("--vision", action="store_true", help="also check that the model reads images")
    a = ap.parse_args()
    ok = check_tools(a)
    if a.vision:
        ok = check_vision(a) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
