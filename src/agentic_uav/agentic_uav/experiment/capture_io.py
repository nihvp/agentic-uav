"""ROS-free parts of the capture step: when the hover is steady, and what gets saved.

A capture directory per scene (experiments/paper/captures/scene_NN/) holds
  rgb_K.png     camera frame K (BGR, as the perception layer sees it)
  depth_K.npz   matching depth image, metres, float32 (key "depth")
  meta.json     planned viewpoint, PX4 pose and camera pose per frame, intrinsics
It holds no scene label or people positions: analysers read only this directory, and
scoring joins it with scenes.json afterwards. eval_true_pose.txt (written by
scripts/capture_scenes.sh from Gazebo) is for evaluation only; analysers must not read it.
"""

import json
import math
import os
from typing import Dict, List, Sequence, Tuple

Sample = Tuple[float, float, float]   # (time s, horizontal+vertical speed m/s, heading rad)


def scene_dir(root: str, scene_id: int) -> str:
    return os.path.join(root, f"scene_{scene_id:02d}")


def is_settled(samples: Sequence[Sample], now: float, window_s: float = 2.0,
               max_speed: float = 0.15, max_yaw_rate: float = 0.03) -> bool:
    """True when every sample in the last window_s is slow and the heading barely moved.

    Needs samples covering the whole window, so a vehicle that just stopped reporting
    does not count as settled.
    """
    recent = [s for s in samples if now - s[0] <= window_s]
    if not recent or now - min(s[0] for s in recent) < 0.8 * window_s:
        return False
    if any(s[1] > max_speed for s in recent):
        return False
    first, last = min(recent), max(recent)
    turn = abs(math.atan2(math.sin(last[2] - first[2]), math.cos(last[2] - first[2])))
    return turn <= max_yaw_rate * max(last[0] - first[0], 1e-3)


def horizontal_offset(planned: Sequence[float], actual: Sequence[float]) -> float:
    return math.hypot(actual[0] - planned[0], actual[1] - planned[1])


def save_frame(directory: str, k: int, bgr, depth) -> Dict[str, str]:
    import cv2
    import numpy as np
    os.makedirs(directory, exist_ok=True)
    rgb_name, depth_name = f"rgb_{k}.png", f"depth_{k}.npz"
    if not cv2.imwrite(os.path.join(directory, rgb_name), bgr):
        raise RuntimeError(f"Could not write {rgb_name} in {directory}.")
    np.savez_compressed(os.path.join(directory, depth_name), depth=np.asarray(depth, np.float32))
    return {"rgb": rgb_name, "depth": depth_name}


def write_meta(directory: str, meta: Dict) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "meta.json")
    with open(path + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(path + ".tmp", path)
    return path


def round_list(values, nd: int = 4) -> List[float]:
    return [round(float(v), nd) for v in values]
