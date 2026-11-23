import math
from types import SimpleNamespace as NS

import numpy as np

from agentic_uav.common.frames import yaw_enu_to_ned
from agentic_uav.experiment.scenes import EMERGENCY, LYING, NORMAL, STANDING, SceneConfig, generate, load, to_json
from agentic_uav.experiment.spawn import BODY_LENGTH, person_sdf, wait_for
from agentic_uav.perception.camera_model import Intrinsics, R_ENU_NED, camera_pose_world
from agentic_uav.perception.detector import CompositeDetector, NullDetector, YoloDetector, make_detector

# The sim camera: 640x360, ~69 deg horizontal field of view, pitched 45 deg down.
FX = 320 / math.tan(math.radians(34.5))
CAM = Intrinsics(FX, FX, 320.0, 180.0, 640, 360)


def project(scene, x, y, z):
    """Pixel of map point (x, y, z) seen from the scene's viewpoint, level, at its yaw."""
    h = yaw_enu_to_ned(math.radians(scene.viewpoint_yaw_deg))
    q = [math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)]
    pos_ned = R_ENU_NED @ np.array(scene.viewpoint)
    t, r = camera_pose_world(pos_ned, q, [0.132, 0.0, 0.261], 0.785)
    p = r.T @ (np.array([x, y, z]) - t)
    return CAM.pixel(p[0] / p[2], p[1] / p[2]), p[2]


def test_balanced_and_deterministic():
    a, b = generate(SceneConfig(seed=3)), generate(SceneConfig(seed=3))
    assert [s.label for s in a] == [s.label for s in b] and a[5].people == b[5].people
    assert sum(s.label == NORMAL for s in a) == sum(s.label == EMERGENCY for s in a) == 22
    assert generate(SceneConfig(seed=4))[0].people != a[0].people


def test_layout_rules():
    cfg = SceneConfig()
    for s in generate(cfg):
        standing = [p for p in s.people if p.posture == STANDING]
        lying = [p for p in s.people if p.posture == LYING]
        assert cfg.crowd_min <= len(standing) <= cfg.crowd_max
        assert len(lying) == (1 if s.label == EMERGENCY else 0)
        for i, p in enumerate(standing):
            assert all(math.hypot(p.x - o.x, p.y - o.y) >= cfg.min_spacing - 1e-6 for o in standing[i + 1:])
        for p in lying:
            assert min(math.hypot(p.x - o.x, p.y - o.y) for o in standing) >= cfg.lying_isolation - 1e-6
        assert len({p.name for p in s.people}) == len(s.people)


def test_everyone_is_in_the_camera_view():
    for s in generate():
        for p in s.people:
            z = 0.9 if p.posture == STANDING else 0.15
            (u, v), depth = project(s, p.x, p.y, z)
            assert depth > 0 and 10 < u < 630 and 10 < v < 350, (s.id, p.name, u, v)


def test_json_round_trip(tmp_path):
    cfg = SceneConfig(n=4, seed=1)
    scenes = generate(cfg)
    path = tmp_path / "scenes.json"
    path.write_text(to_json(scenes, cfg))
    assert load(str(path)) == scenes


def test_lying_sdf_is_centred_on_the_person():
    p = generate()[0].people[0]
    lying = NS(name="x", posture=LYING, x=10.0, y=5.0, yaw=0.0)
    sdf = person_sdf(lying)
    pose = [float(v) for v in sdf.split("<pose>")[1].split("</pose>")[0].split()]
    # yaw 0: rolling +90 deg about x points the body (feet -> head) along -y, so the feet
    # sit half a body length towards +y of the centre.
    assert math.isclose(pose[0], 10.0) and math.isclose(pose[1], 5.0 + BODY_LENGTH / 2)
    assert math.isclose(pose[3], math.pi / 2, abs_tol=1e-3)
    assert "<static>true</static>" in sdf and '"' not in sdf
    assert f"<name>{p.name}</name>" in person_sdf(p)


def test_wait_for_models_that_appear_later():
    t = [0.0]
    worlds = iter([["exp_a"], ["exp_a"], ["exp_a", "exp_b"]])
    tick = lambda s: t.__setitem__(0, t[0] + s)
    assert wait_for(["exp_a", "exp_b"], 10, lambda: next(worlds), clock=lambda: t[0], sleep=tick) == []
    assert wait_for(["exp_c"], 5, lambda: [], clock=lambda: t[0], sleep=tick) == ["exp_c"]


class FakeYolo:
    def predict(self, img, **kw):
        boxes = NS(xyxy=NS(tolist=lambda: [[1.2, 2, 30, 60.6], [5, 5, 9, 9]]),
                   conf=NS(tolist=lambda: [0.81234, 0.9]), cls=NS(tolist=lambda: [0, 2]))
        return [NS(names={0: "person", 2: "car"}, boxes=boxes)]


def test_yolo_keeps_only_requested_classes():
    det = YoloDetector(backend=FakeYolo()).detect(np.zeros((10, 10, 3), np.uint8))
    assert det == [{"bbox": [1, 2, 30, 61], "label": "person", "confidence": 0.812, "source": "yolo"}]


def test_make_detector_composites():
    assert isinstance(make_detector("none"), NullDetector)
    d = make_detector("color+yolo")
    assert isinstance(d, CompositeDetector) and isinstance(d.detectors[1], YoloDetector)


def test_settled_needs_a_full_steady_window():
    from agentic_uav.experiment.capture_io import is_settled
    steady = [(t / 10, 0.05, 1.0) for t in range(0, 31)]           # 3 s, slow, fixed heading
    assert is_settled(steady, 3.0)
    assert not is_settled(steady[:5], 0.4)                          # too little history
    assert not is_settled(steady, 10.0)                             # stale: nothing recent
    moving = steady[:-3] + [(2.8, 0.5, 1.0), (2.9, 0.05, 1.0), (3.0, 0.05, 1.0)]
    assert not is_settled(moving, 3.0)
    turning = [(t / 10, 0.05, 1.0 + 0.05 * t / 10) for t in range(0, 31)]   # 0.05 rad/s
    assert not is_settled(turning, 3.0)


def test_capture_meta_round_trip(tmp_path):
    import json
    from agentic_uav.experiment.capture_io import horizontal_offset, scene_dir, write_meta
    d = scene_dir(str(tmp_path), 7)
    assert d.endswith("scene_07")
    write_meta(d, {"scene_id": 7})
    assert json.loads((tmp_path / "scene_07" / "meta.json").read_text()) == {"scene_id": 7}
    assert math.isclose(horizontal_offset((0, 0, 6), (3, 4, 5)), 5.0)
