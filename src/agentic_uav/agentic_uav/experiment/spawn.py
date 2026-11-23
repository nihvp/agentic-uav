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
import time
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


def parse_model_list(out: str) -> List[str]:
    """Model names from `gz model --list`. The world always has models (ground, the drone),
    so an empty list means the query failed (e.g. timed out under load), not an empty world."""
    names = [line.strip()[2:].strip() for line in out.splitlines() if line.strip().startswith("- ")]
    if not names:
        raise RuntimeError(f"gz model --list returned no models: {out.strip()[:200]!r}")
    return names


def list_models(attempts: int = 3) -> List[str]:
    for i in range(attempts):
        try:
            return parse_model_list(_gz(["model", "--list"], 30))
        except (RuntimeError, subprocess.TimeoutExpired):
            if i == attempts - 1:
                raise
            time.sleep(2.0)
    return []


def experiment_models(names: List[str]) -> List[str]:
    return [n for n in names if n.startswith(PREFIX)]


def clear(world: str, attempts: int = 3) -> int:
    """Remove every experiment model; raise unless the world is verified free of them."""
    removed = 0
    for _ in range(attempts):
        names = experiment_models(list_models())
        if not names:
            return removed
        for n in names:
            _gz(["service", "-s", f"/world/{world}/remove", "--reqtype", "gz.msgs.Entity",
                 "--reptype", "gz.msgs.Boolean", "--timeout", "10000",
                 "--req", f'name: "{n}" type: MODEL'], 20)
        removed += len(names)
        time.sleep(2.0)      # removal is queued like creation
    left = experiment_models(list_models())
    if left:
        raise RuntimeError(f"Could not remove {len(left)} experiment models: {left[:5]}")
    return removed


def spawn(world: str, people: List[Person], timeout_ms: int = 60000):
    # The first spawn downloads the model from Fuel (cached afterwards), hence the long timeout.
    for p in people:
        sdf = person_sdf(p).replace('"', '\\"')
        out = _gz(["service", "-s", f"/world/{world}/create", "--reqtype", "gz.msgs.EntityFactory",
                   "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout_ms),
                   "--req", f'sdf: "{sdf}"'], timeout_ms / 1000 + 10)
        if "data: true" not in out:
            print(f"[spawn] {p.name}: unexpected reply: {out.strip()[:200]}", file=sys.stderr)


def wait_for(names: List[str], timeout_s: float, list_fn=list_models, poll_s: float = 2.0,
             clock=time.monotonic, sleep=time.sleep) -> List[str]:
    """Wait until every name is in the world; return the ones still missing at the timeout.

    Gazebo's create service only queues the request and replies true at once; the
    simulation thread builds the model later (and the first Fuel include blocks it while
    the model downloads), so models appear some time after the replies.
    """
    deadline = clock() + timeout_s
    while True:
        try:
            present = set(list_fn())
        except (RuntimeError, subprocess.TimeoutExpired):
            present = set()          # query failed this time; poll again
        missing = [n for n in names if n not in present]
        if not missing or clock() >= deadline:
            return missing
        sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default="/experiments/paper/scenes.json")
    ap.add_argument("--scene", type=int, help="scene id to spawn (clears the previous one)")
    ap.add_argument("--clear", action="store_true", help="only remove experiment people")
    ap.add_argument("--world", default="search_and_rescue")
    ap.add_argument("--wait", type=float, default=180.0,
                    help="seconds to wait for the models to appear (first run downloads from Fuel)")
    a = ap.parse_args()

    try:
        removed = clear(a.world)
    except RuntimeError as e:
        print(f"[spawn] {e}", file=sys.stderr)
        return 1
    print(f"[spawn] removed {removed} experiment models")
    if a.clear or a.scene is None:
        return 0
    scene = next((s for s in load(a.scenes) if s.id == a.scene), None)
    if scene is None:
        print(f"[spawn] no scene {a.scene} in {a.scenes}", file=sys.stderr)
        return 1
    spawn(a.world, scene.people)
    missing = wait_for([p.name for p in scene.people], a.wait)
    # Exactly this scene's people: a leftover from an earlier scene would join the sample.
    extra = sorted(set(experiment_models(list_models())) - {p.name for p in scene.people})
    print(f"[spawn] scene {scene.id} ({scene.label}): {len(scene.people) - len(missing)}/{len(scene.people)} "
          f"people in the world; viewpoint {scene.viewpoint} yaw {scene.viewpoint_yaw_deg} deg")
    if missing or extra:
        print(f"[spawn] missing: {missing}; not from this scene: {extra}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
