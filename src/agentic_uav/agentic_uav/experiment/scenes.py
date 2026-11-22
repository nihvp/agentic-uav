"""Crowd scenes for the paper reproduction (docs/PAPER_REPRODUCTION.md).

Each scene is one sample: a crowd of standing people and, in emergency scenes, one
person lying on the ground away from the crowd. Scenes come from a fixed seed so the
same 44 can be regenerated anywhere; the written file is the experiment's ground truth.
It is read only by the experiment harness (spawner, scorer), never by an agent layer.

Layout. The drone hovers at a viewpoint facing the scene's look-at point L, the ground
point on the camera's centre ray (camera pitched 45 deg down, so L is `altitude` metres
ahead). People are placed in a frame aligned with the view (u forward, v left) around L,
inside the camera footprint, so every person is in view.

Standard library only: this also runs in the sim container, which has no numpy.

  python3 -m agentic_uav.experiment.scenes --out experiments/paper/scenes.json
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

NORMAL = "normal"
EMERGENCY = "emergency"
STANDING = "standing"
LYING = "lying"


@dataclass
class Person:
    name: str
    posture: str          # standing | lying
    x: float
    y: float
    yaw: float            # rad, ENU map frame


@dataclass
class Scene:
    id: int
    label: str            # normal | emergency
    look_at: Tuple[float, float]
    viewpoint: Tuple[float, float, float]   # x, y, z (ENU, m)
    viewpoint_yaw_deg: float                # map frame, 0 = east
    people: List[Person] = field(default_factory=list)

    @property
    def num_standing(self) -> int:
        return sum(p.posture == STANDING for p in self.people)

    @property
    def num_lying(self) -> int:
        return sum(p.posture == LYING for p in self.people)


@dataclass
class SceneConfig:
    n: int = 44
    seed: int = 0
    # Scene site, clear of the search_and_rescue world's objects (all within 15 m of home).
    site: Tuple[float, float] = (30.0, -20.0)
    altitude: float = 6.0
    crowd_min: int = 6
    crowd_max: int = 12
    # Crowd: disc of this radius, centred this far beyond L along the view.
    crowd_radius: float = 2.0
    crowd_ahead: float = 1.5
    # Standing people at least this far apart (m).
    min_spacing: float = 0.8
    # Lying person, view frame relative to L: nearer than L, off to one side. Kept inside
    # the camera footprint (about 69 x 42 deg) with a few degrees of margin.
    lying_u: Tuple[float, float] = (-2.5, -1.5)
    lying_v_abs: Tuple[float, float] = (2.5, 3.5)
    # ... and at least this far from every standing person ("isolated from the crowd").
    lying_isolation: float = 2.0


def _view_to_map(look_at, yaw, u, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return look_at[0] + u * c - v * s, look_at[1] + u * s + v * c


def _crowd(rng: random.Random, cfg: SceneConfig, look_at, yaw, count: int, prefix: str) -> List[Person]:
    people: List[Person] = []
    for _ in range(count * 50):
        if len(people) == count:
            break
        r = cfg.crowd_radius * math.sqrt(rng.random())
        a = rng.uniform(0, 2 * math.pi)
        x, y = _view_to_map(look_at, yaw, cfg.crowd_ahead + r * math.cos(a), r * math.sin(a))
        if all(math.hypot(x - p.x, y - p.y) >= cfg.min_spacing for p in people):
            people.append(Person(f"{prefix}_p{len(people) + 1}", STANDING, round(x, 3), round(y, 3),
                                 round(rng.uniform(-math.pi, math.pi), 3)))
    if len(people) < count:
        raise RuntimeError(f"Could not place {count} people {cfg.min_spacing} m apart; widen crowd_radius.")
    return people


def _lying(rng: random.Random, cfg: SceneConfig, look_at, yaw, crowd: Sequence[Person], prefix: str) -> Person:
    for _ in range(500):
        u = rng.uniform(*cfg.lying_u)
        v = rng.choice((-1, 1)) * rng.uniform(*cfg.lying_v_abs)
        x, y = _view_to_map(look_at, yaw, u, v)
        if all(math.hypot(x - p.x, y - p.y) >= cfg.lying_isolation for p in crowd):
            return Person(f"{prefix}_lying", LYING, round(x, 3), round(y, 3),
                          round(rng.uniform(-math.pi, math.pi), 3))
    raise RuntimeError("Could not place the lying person away from the crowd.")


def generate(cfg: Optional[SceneConfig] = None) -> List[Scene]:
    cfg = cfg or SceneConfig()
    if cfg.n % 2:
        raise ValueError("n must be even for a balanced set.")
    rng = random.Random(cfg.seed)
    labels = [NORMAL] * (cfg.n // 2) + [EMERGENCY] * (cfg.n // 2)
    rng.shuffle(labels)
    scenes = []
    for i, label in enumerate(labels, start=1):
        yaw_deg = round(rng.uniform(0, 360), 1)
        yaw = math.radians(yaw_deg)
        # Small jitter so samples are not one fixed picture.
        look_at = (round(cfg.site[0] + rng.uniform(-2, 2), 3), round(cfg.site[1] + rng.uniform(-2, 2), 3))
        vx, vy = _view_to_map(look_at, yaw, -cfg.altitude, 0.0)   # 45 deg pitch: L is altitude ahead
        prefix = f"exp{i:02d}"
        people = _crowd(rng, cfg, look_at, yaw, rng.randint(cfg.crowd_min, cfg.crowd_max), prefix)
        if label == EMERGENCY:
            people.append(_lying(rng, cfg, look_at, yaw, people, prefix))
        scenes.append(Scene(i, label, look_at, (round(vx, 3), round(vy, 3), cfg.altitude), yaw_deg, people))
    return scenes


def to_json(scenes: Sequence[Scene], cfg: SceneConfig) -> str:
    return json.dumps({"config": asdict(cfg), "scenes": [
        {**asdict(s), "num_standing": s.num_standing, "num_lying": s.num_lying} for s in scenes]}, indent=1)


def load(path: str) -> List[Scene]:
    with open(path) as f:
        data = json.load(f)
    return [Scene(s["id"], s["label"], tuple(s["look_at"]), tuple(s["viewpoint"]), s["viewpoint_yaw_deg"],
                  [Person(**p) for p in s["people"]]) for s in data["scenes"]]


def main():
    ap = argparse.ArgumentParser(description="Write the experiment's scene set (ground truth).")
    ap.add_argument("--out", default="experiments/paper/scenes.json")
    ap.add_argument("--n", type=int, default=44)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = SceneConfig(n=a.n, seed=a.seed)
    scenes = generate(cfg)
    with open(a.out, "w") as f:
        f.write(to_json(scenes, cfg) + "\n")
    print(f"Wrote {len(scenes)} scenes ({sum(s.label == EMERGENCY for s in scenes)} emergency) to {a.out}")


if __name__ == "__main__":
    main()
