"""One-shot check that the configured LLM server answers with a tool call.

Run inside the agent container (no ROS or simulator involved):
  python3 -m agentic_uav.reasoning.llm_check [--provider ollama] [--model qwen3:14b]
"""

import argparse
import os
import sys
import time

from agentic_uav.reasoning.llm import Message, ToolSpec, make_provider


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3:14b"))
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--timeout", type=float, default=300.0, help="first call also loads the model into VRAM")
    a = ap.parse_args()

    tools = [ToolSpec("takeoff", "Take off and hover at the given altitude.",
                      {"type": "object", "properties": {"altitude": {"type": "number", "description": "metres"}},
                       "required": ["altitude"]})]
    provider = make_provider(a.provider, model=a.model, base_url=a.base_url, timeout_s=a.timeout)
    t0 = time.monotonic()
    resp = provider.complete("You control a drone through tools. Act by calling a tool.",
                             [Message("user", "Take off to 5 metres.")], tools)
    dt = time.monotonic() - t0
    print(f"model={a.model} took {dt:.1f} s")
    print(f"text: {resp.text!r}")
    print(f"tool calls: {[(c.name, c.arguments) for c in resp.tool_calls]}")
    ok = any(c.name == "takeoff" and abs(float(c.arguments.get("altitude", 0)) - 5.0) < 0.01
             for c in resp.tool_calls)
    print("PASS" if ok else "FAIL: expected takeoff(altitude=5)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
