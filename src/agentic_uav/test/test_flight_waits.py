import pytest

from agentic_uav.reasoning.flight_waits import FlightWaitError, wait_landed, wait_takeoff


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def timeline(clock, states):
    """Telemetry getter that plays [(from_time, telemetry), ...]."""
    def get():
        current = None
        for start, tel in states:
            if clock.t >= start:
                current = tel
        return current
    return get


def tel(state, z=0.0, armed=True, nav="OFFBOARD"):
    return {"controller_state": state, "position": [0.0, 0.0, z], "armed": armed, "px4_nav_state": nav}


def test_takeoff_waits_for_holding_at_altitude():
    clock = FakeClock()
    get = timeline(clock, [(0, tel("STARTING")), (2, tel("TAKING_OFF", 2.0)), (6, tel("HOLDING", 4.9))])
    t = wait_takeoff(get, 5.0, timeout=30, clock=clock, sleep=clock.sleep)
    assert t["controller_state"] == "HOLDING" and clock.t >= 6


def test_takeoff_ignores_stale_holding_at_old_altitude():
    # Already hovering at 3 m: the first HOLDING sample is the old state, not the new climb.
    clock = FakeClock()
    get = timeline(clock, [(0, tel("HOLDING", 3.0)), (0.5, tel("TAKING_OFF", 3.5)), (4, tel("HOLDING", 8.0))])
    t = wait_takeoff(get, 8.0, timeout=30, clock=clock, sleep=clock.sleep)
    assert t["position"][2] == 8.0


def test_takeoff_reports_refused_arming():
    clock = FakeClock()
    get = timeline(clock, [(0, tel("STARTING", armed=False)), (5, tel("IDLE", armed=False))])
    with pytest.raises(FlightWaitError, match="did not arm"):
        wait_takeoff(get, 5.0, timeout=30, clock=clock, sleep=clock.sleep)


def test_takeoff_reports_failsafe_and_timeout():
    clock = FakeClock()
    get = timeline(clock, [(0, tel("TAKING_OFF", 1.0)), (3, tel("EXTERNAL", 1.5, nav="AUTO_LAND"))])
    with pytest.raises(FlightWaitError, match="AUTO_LAND"):
        wait_takeoff(get, 5.0, timeout=30, clock=clock, sleep=clock.sleep)

    clock = FakeClock()
    with pytest.raises(FlightWaitError, match="within 10 s"):
        wait_takeoff(timeline(clock, [(0, tel("TAKING_OFF", 1.0))]), 5.0, timeout=10,
                     clock=clock, sleep=clock.sleep)


def test_landing_waits_for_disarm():
    clock = FakeClock()
    get = timeline(clock, [(0, tel("HANDED_OFF", 5.0, nav="AUTO_RTL")), (20, tel("IDLE", 0.0, armed=False))])
    t = wait_landed(get, timeout=60, what="Return to launch", clock=clock, sleep=clock.sleep)
    assert not t["armed"] and clock.t >= 20
