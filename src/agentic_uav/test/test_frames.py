import math

from agentic_uav.common.frames import enu_to_ned, ned_to_enu, yaw_enu_to_ned, yaw_ned_to_enu


def test_position_round_trip():
    p = (1.0, 2.0, 3.0)
    assert enu_to_ned(p) == (2.0, 1.0, -3.0)
    assert ned_to_enu(enu_to_ned(p)) == p


def test_yaw_east_is_ned_quarter_turn():
    assert math.isclose(yaw_enu_to_ned(0.0), math.pi / 2)
    assert math.isclose(yaw_ned_to_enu(yaw_enu_to_ned(0.7)), 0.7)
