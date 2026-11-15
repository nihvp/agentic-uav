"""Helpers for the scan tool: turn in place and collect what perception reports.

Each LLM request costs quota (the free tier allows ~20 a day), so a 360-degree look
is one tool call instead of a turn + get_scene_graph pair per heading. Everything
returned still comes from the perception layer's scene graph; nothing is inferred here.
"""

from typing import Any, Dict, List


def scan_headings(start_deg: float, step_deg: float = 60.0) -> List[float]:
    """Headings for one full turn starting after the current one, ending back at the start."""
    step = max(15.0, min(180.0, abs(step_deg)))
    n = max(1, round(360.0 / step))
    return [round((start_deg + step * (i + 1)) % 360.0, 1) for i in range(n)]


def compact_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scene nodes trimmed to what the LLM needs, confirmed first."""
    out = [{"id": n["id"], "label": n["label"], "position": n["position"],
            "existence_prob": n["existence_prob"],
            "confirmed": "state:confirmed" in n.get("attributes", [])} for n in nodes]
    return sorted(out, key=lambda n: (not n["confirmed"], -n["existence_prob"]))


def merge_sightings(seen: Dict[int, Dict[str, Any]], nodes: List[Dict[str, Any]], heading_deg: float):
    """Record each node by id, keeping its latest position and best existence_prob."""
    for n in compact_nodes(nodes):
        prev = seen.get(n["id"])
        entry = dict(n, seen_at_heading_deg=heading_deg)
        if prev is not None:
            entry["existence_prob"] = max(prev["existence_prob"], n["existence_prob"])
            entry["confirmed"] = prev["confirmed"] or n["confirmed"]
        seen[n["id"]] = entry
