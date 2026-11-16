"""Mission sighting log: confirmed perception sightings remembered for the whole mission.

The scene graph forgets an object a few seconds after it leaves the camera view (and
gives it a new id when it comes back), which is right for perception but leaves the
LLM to recall positions from a long history. This log keeps every confirmed sighting,
merges repeats of the same object, and answers "what is near this point". It only
holds what perception reported; it never adds objects or reads simulator state.
"""

import math
from typing import Any, Dict, List

# Same-label sightings closer than this (m, horizontal) are taken as one object. Large
# objects (the landing pad) are placed up to ~3 m apart from different viewpoints.
MERGE_DISTANCE = {"landing_zone": 4.0}
DEFAULT_MERGE_DISTANCE = 2.5


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class SightingLog:
    def __init__(self):
        self._objects: List[Dict[str, Any]] = []

    def reset(self):
        self._objects = []

    def add(self, nodes: List[Dict[str, Any]]):
        """Record compacted scene nodes (see scan.compact_nodes); only confirmed ones count."""
        for n in nodes:
            if not n.get("confirmed"):
                continue
            gate = MERGE_DISTANCE.get(n["label"], DEFAULT_MERGE_DISTANCE)
            match = min((o for o in self._objects
                         if o["label"] == n["label"] and _xy_dist(o["position"], n["position"]) < gate),
                        key=lambda o: _xy_dist(o["position"], n["position"]), default=None)
            if match is None:
                self._objects.append({"ref": f"{n['label']}-{sum(o['label'] == n['label'] for o in self._objects) + 1}",
                                      "label": n["label"], "position": [round(v, 2) for v in n["position"]],
                                      "times_seen": 1})
            else:
                k = match["times_seen"]
                match["position"] = [round((p * k + q) / (k + 1), 2) for p, q in zip(match["position"], n["position"])]
                match["times_seen"] = k + 1

    def known(self) -> List[Dict[str, Any]]:
        return [dict(o) for o in self._objects]

    def near(self, position, radius: float = 1.5) -> List[Dict[str, Any]]:
        """Known objects within radius (m, horizontal) of position, nearest first."""
        hits = [(d, o) for o in self._objects if (d := _xy_dist(o["position"], position)) <= radius]
        return [{"ref": o["ref"], "label": o["label"], "horizontal_distance_m": round(d, 2)}
                for d, o in sorted(hits, key=lambda h: h[0])]
