"""Frame conversions between ROS (ENU / FLU) and PX4 (NED / FRD).

ROS map frame: x east, y north, z up. Yaw 0 = east, counter-clockwise positive.
PX4 local frame: x north, y east, z down. Yaw 0 = north, clockwise positive.
"""

import math
from typing import Sequence, Tuple

Vec3 = Tuple[float, float, float]


def enu_to_ned(v: Sequence[float]) -> Vec3:
    return (float(v[1]), float(v[0]), -float(v[2]))


def ned_to_enu(v: Sequence[float]) -> Vec3:
    return (float(v[1]), float(v[0]), -float(v[2]))


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_enu_to_ned(yaw: float) -> float:
    return wrap_pi(math.pi / 2.0 - yaw)


def yaw_ned_to_enu(yaw: float) -> float:
    return wrap_pi(math.pi / 2.0 - yaw)
