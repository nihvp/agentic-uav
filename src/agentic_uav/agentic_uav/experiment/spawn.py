"""Put one experiment scene's people into the running Gazebo world (sim container).

Scene setup between samples, done by the experiment harness: it removes the previous
scene's people (models named exp*) and spawns the requested scene's through Gazebo's
create/remove services. Nothing moves while a sample is captured. Standard library and
the `gz` CLI only.

  docker compose exec sim python3 -m agentic_uav.experiment.spawn --scene 3
  docker compose exec sim python3 -m agentic_uav.experiment.spawn --clear
"""

import argparse
import math
import subprocess
import sys
from typing import List

from agentic_uav.experiment.scenes import LYING, Person, load

PERSON_URI = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing person"
# The standing model's origin is at its feet. Lying = rolled 90 deg onto its back; the
# body then extends ~BODY_LENGTH from the feet, and LYING_Z lifts it clear of the ground.
BODY_LENGTH = 1.7
LYING_Z = 0.15
PREFIX = "exp"


def person_sdf(p: Person, uri: str = PERSON_URI) -> str:
    """One-line SDF <include> for a static person whose body is centred on (p.x, p.y)."""
    if p.posture == LYING:
        # Roll +90 deg about x sends the model's +z (feet -> head) to -y, then yaw turns it.
        dx, dy = math.sin(p.yaw), -math.cos(p.yaw)          # head direction in the map
        x, y = p.x - 0.5 * BODY_LENGTH * dx, p.y - 0.5 * BODY_LENGTH * dy
        pose = f"{x:.3f} {y:.3f} {LYING_Z:.3f} {math.pi / 2:.4f} 0 {p.yaw:.4f}"
    else:
        pose = f"{p.x:.3f} {p.y:.3f} 0 0 0 {p.yaw:.4f}"
    return (f"<sdf version='1.9'><include><uri>{uri}</uri><name>{p.name}</name>"
            f"<static>true</static><pose>{pose}</pose></include></sdf>")


def _gz(args: List[str], timeout_s: float) -> str:
    res = subprocess.run(["gz", *args], capture_output=True, text=True, timeout=timeout_s)
    return res.stdout + res.stderr


def list_models() -> List[str]:
    out = _gz(["model", "--list"], 30)
    return [line.strip()[2:].strip() for line in out.splitlines() if line.strip().startswith("- ")]


def clear(world: str) -> int:
    names = [n for n in list_models() if n.startswith(PREFIX)]
    for n in names:
        _gz(["service", "-s", f"/world/{world}/remove", "--reqtype", "gz.msgs.Entity",
             "--reptype", "gz.msgs.Boolean", "--timeout", "10000",
             "--req", f'name: "{n}" type: MODEL'], 20)
    return len(names)


def spawn(world: str, people: List[Person], timeout_ms: int = 60000):
    # The first spawn downloads the model from Fuel (cached afterwards), hence the long timeout.
    for p in people:
        sdf = person_sdf(p).replace('"', '\\"')
        out = _gz(["service", "-s", f"/world/{world}/create", "--reqtype", "gz.msgs.EntityFactory",
                   "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout_ms),
                   "--req", f'sdf: "{sdf}"'], timeout_ms / 1000 + 10)
        if "data: true" not in out:
            print(f"[spawn] {p.name}: unexpected reply: {out.strip()[:200]}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default="/experiments/paper/scenes.json")
    ap.add_argument("--scene", type=int, help="scene id to spawn (clears the previous one)")
    ap.add_argument("--clear", action="store_true", help="only remove experiment people")
    ap.add_argument("--world", default="search_and_rescue")
    a = ap.parse_args()

    removed = clear(a.world)
    print(f"[spawn] removed {removed} experiment models")
    if a.clear or a.scene is None:
        return 0
    scene = next((s for s in load(a.scenes) if s.id == a.scene), None)
    if scene is None:
        print(f"[spawn] no scene {a.scene} in {a.scenes}", file=sys.stderr)
        return 1
    spawn(a.world, scene.people)
    present = set(list_models())
    missing = [p.name for p in scene.people if p.name not in present]
    print(f"[spawn] scene {scene.id} ({scene.label}): {len(scene.people) - len(missing)}/{len(scene.people)} "
          f"people in the world; viewpoint {scene.viewpoint} yaw {scene.viewpoint_yaw_deg} deg")
    if missing:
        print(f"[spawn] missing: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
