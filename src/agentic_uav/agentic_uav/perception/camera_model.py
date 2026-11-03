"""Camera geometry: pinhole intrinsics and the camera pose in the ENU map frame.

Frames:
  NED / FRD     PX4 local position and vehicle_attitude (q rotates FRD -> NED)
  ENU / FLU     ROS map and body frames
  camera_link   x forward, mounted on the body at `mount_xyz` (FLU), pitched down by
                `mount_pitch` rad (SDF pose convention: positive pitch = nose down)
  optical       x right, y down, z forward (what image pixels and depth refer to)
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
R_FRD_FLU = np.diag([1.0, -1.0, -1.0])
R_CAMLINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, msg) -> "Intrinsics":
        return cls(msg.k[0], msg.k[4], msg.k[2], msg.k[5], msg.width, msg.height)

    def ray(self, u: float, v: float) -> Tuple[float, float]:
        """Normalised image coordinates (X/Z, Y/Z) of pixel (u, v)."""
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def pixel(self, x: float, y: float) -> Tuple[float, float]:
        return self.cx + self.fx * x, self.cy + self.fy * y


def quat_to_rot(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix from a unit quaternion [w, x, y, z]."""
    w, x, y, z = (float(c) for c in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def camera_pose_world(pos_ned: Sequence[float], q_ned_frd: Sequence[float],
                      mount_xyz: Sequence[float], mount_pitch: float) -> Tuple[np.ndarray, np.ndarray]:
    """(t, R) with p_world = R @ p_optical + t, world = ENU map frame."""
    r_world_flu = R_ENU_NED @ quat_to_rot(q_ned_frd) @ R_FRD_FLU
    r_world_opt = r_world_flu @ _rot_y(mount_pitch) @ R_CAMLINK_OPTICAL
    t_world = R_ENU_NED @ np.asarray(pos_ned, dtype=float) + r_world_flu @ np.asarray(mount_xyz, dtype=float)
    return t_world, r_world_opt


def bbox_point(bbox: Sequence[float], depth: np.ndarray, rgb: Intrinsics,
               depth_k: Intrinsics) -> Optional[np.ndarray]:
    """Median 3D point (optical frame, m) of the valid depth pixels inside an RGB bbox.

    Every depth pixel in the box is back-projected and the per-axis median taken, which
    stays on the object for large or slanted shapes (a ground pad seen obliquely) where a
    single depth at the box centre would not. The RGB and depth sensors share an optical
    centre but differ in resolution and field of view, so the bbox corners are mapped
    through the normalised image plane. Returns None when the box holds no valid depth.
    """
    u1, v1 = depth_k.pixel(*rgb.ray(bbox[0], bbox[1]))
    u2, v2 = depth_k.pixel(*rgb.ray(bbox[2], bbox[3]))
    h, w = depth.shape[:2]
    x1, x2 = max(0, int(u1)), min(w, int(np.ceil(u2)))
    y1, y2 = max(0, int(v1)), min(h, int(np.ceil(v2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = depth[y1:y2, x1:x2]
    ok = np.isfinite(crop) & (crop > 0.1)
    if not ok.any():
        return None
    vs, us = np.nonzero(ok)
    z = crop[ok].astype(float)
    x = (us + x1 - depth_k.cx) / depth_k.fx * z
    y = (vs + y1 - depth_k.cy) / depth_k.fy * z
    return np.array([np.median(x), np.median(y), np.median(z)])
