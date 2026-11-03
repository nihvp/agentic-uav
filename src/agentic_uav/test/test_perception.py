import math

import cv2
import numpy as np

from agentic_uav.perception.camera_model import Intrinsics, bbox_point, camera_pose_world
from agentic_uav.perception.detector import ColorDetector
from agentic_uav.perception.scene_graph_builder import SceneGraphBuilder

LEVEL_NORTH = (1.0, 0.0, 0.0, 0.0)            # PX4 q (w, x, y, z): level, nose north
LEVEL_EAST = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
PITCH = math.pi / 4
RGB = Intrinsics(fx=320.0, fy=320.0, cx=320.0, cy=180.0, width=640, height=360)
DEPTH = Intrinsics(fx=300.0, fy=300.0, cx=320.0, cy=240.0, width=640, height=480)


def project(p_world, t, r, k):
    p = r.T @ (np.asarray(p_world) - t)
    return k.cx + k.fx * p[0] / p[2], k.cy + k.fy * p[1] / p[2], p[2]


def test_optical_axis_hits_ground_ahead_of_pitched_camera():
    # 5 m up, nose north, camera pitched 45 deg down: centre ray meets the ground 5 m north.
    t, r = camera_pose_world((0.0, 0.0, -5.0), LEVEL_NORTH, (0.0, 0.0, 0.0), PITCH)
    hit = r @ np.array([0.0, 0.0, 5.0 * math.sqrt(2)]) + t
    assert np.allclose(hit, (0.0, 5.0, 0.0), atol=1e-9)


def test_heading_and_mount_offset_rotate_with_body():
    t, r = camera_pose_world((0.0, 0.0, -5.0), LEVEL_EAST, (0.2, 0.0, 0.0), PITCH)
    assert np.allclose(t, (0.2, 0.0, 5.0))                    # mount offset now points east
    forward = r @ np.array([0.0, 0.0, 1.0])
    assert forward[0] > 0.7 and abs(forward[1]) < 1e-9 and forward[2] < -0.7
    right = r @ np.array([1.0, 0.0, 0.0])                     # image right = south when facing east
    assert np.allclose(right, (0.0, -1.0, 0.0))


def test_bbox_point_maps_rgb_box_onto_depth_image():
    depth = np.full((480, 640), np.inf, np.float32)
    depth[200:280, 280:360] = 7.0      # centred patch in the depth image
    # The same rays in the RGB camera: 40 px around the centre in both (equal focal ratio).
    box = [320 - 40 * 320 / 300, 180 - 40 * 320 / 300, 320 + 40 * 320 / 300, 180 + 40 * 320 / 300]
    p = bbox_point(box, depth, RGB, DEPTH)
    assert p[2] == 7.0 and abs(p[0]) < 0.02 and abs(p[1]) < 0.02
    assert bbox_point([0, 0, 10, 10], depth, RGB, DEPTH) is None   # only invalid depth there


def ground_depth_image(t, r, k):
    """Depth image (along the optical axis) of the z=0 ground plane seen from pose (t, r)."""
    us, vs = np.meshgrid(np.arange(k.width) + 0.5, np.arange(k.height) + 0.5)
    rays = np.stack([(us - k.cx) / k.fx, (vs - k.cy) / k.fy, np.ones_like(us)], axis=-1)
    world_dz = rays @ r[2]                       # world z change per unit optical depth
    with np.errstate(divide="ignore"):
        depth = np.where(world_dz < 0, -t[2] / world_dz, np.inf)
    return depth.astype(np.float32)


def test_slanted_ground_pad_is_placed_on_the_ground():
    # Flat 2 m-radius pad seen obliquely: its box's median point must stay on the ground.
    t, r = camera_pose_world((0.0, 0.0, -5.0), LEVEL_NORTH, (0.0, 0.0, 0.0), PITCH)
    pad = np.array([0.0, 6.0, 0.0])
    corners = [project(pad + d, t, r, RGB) for d in ((-2, -2, 0), (2, -2, 0), (-2, 2, 0), (2, 2, 0))]
    us, vs = [c[0] for c in corners], [c[1] for c in corners]
    box = [min(us), min(vs), max(us), max(vs)]
    p_world = r @ bbox_point(box, ground_depth_image(t, r, DEPTH), RGB, DEPTH) + t
    assert abs(p_world[2]) < 0.05 and np.linalg.norm(p_world[:2] - pad[:2]) < 0.5


def test_color_detector_finds_world_colours_and_ignores_grey():
    img = np.full((360, 640, 3), 150, np.uint8)               # grey ground
    img[100:160, 100:130] = cv2.cvtColor(np.uint8([[[16, 172, 120]]]), cv2.COLOR_HSV2BGR)  # survivor as rendered
    img[200:260, 400:480] = (13, 38, 242)                     # fire red (hue ~4)
    img[300:340, 200:240] = cv2.cvtColor(np.uint8([[[7, 205, 104]]]), cv2.COLOR_HSV2BGR)  # fire as rendered
    img[50:90, 500:600] = (230, 102, 25)                      # landing pad blue
    found = ColorDetector().detect(img)
    assert sorted(d["label"] for d in found) == ["fire_hazard", "fire_hazard", "landing_zone", "survivor"]
    dets = {d["label"]: d for d in found}
    assert dets["survivor"]["bbox"] == [100, 100, 130, 160]
    assert all(d["source"] == "color" for d in dets.values())
    assert ColorDetector().detect(np.full((360, 640, 3), 150, np.uint8)) == []


def test_detection_lands_at_true_world_position_and_decays_out_of_view():
    survivor = np.array([8.0, 4.0, 0.7])
    t, r = camera_pose_world((-1.0, 8.0, -5.0), LEVEL_NORTH, (0.13, 0.0, 0.26), PITCH)
    u, v, z = project(survivor, t, r, RGB)
    builder = SceneGraphBuilder(RGB.fx, RGB.fy, RGB.cx, RGB.cy)
    point_cam = r.T @ (survivor - t)
    det = {"bbox": [u - 10, v - 20, u + 10, v + 20], "label": "survivor", "confidence": 0.8,
           "source": "color", "point_cam": point_cam}
    for _ in range(5):
        nodes, _ = builder.process_detections([dict(det)], t, r)
    assert len(nodes) == 1
    assert np.linalg.norm(np.array(nodes[0]["position"]) - survivor) < 1e-3
    assert "state:confirmed" in nodes[0]["attributes"] and "detector:color" in nodes[0]["attributes"]

    for _ in range(20):
        nodes, _ = builder.process_detections([], t, r)
    assert nodes == []


def test_detection_without_depth_is_dropped():
    builder = SceneGraphBuilder()
    t, r = camera_pose_world((0.0, 0.0, -5.0), LEVEL_NORTH, (0.0, 0.0, 0.0), PITCH)
    nodes, _ = builder.process_detections(
        [{"bbox": [0, 0, 10, 10], "label": "survivor", "confidence": 0.9, "point_cam": None}], t, r)
    assert nodes == []
