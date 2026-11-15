"""Wait for a flight command's real outcome in the action layer's telemetry.

FlightCommand returns as soon as the action layer accepts a command; the vehicle
takes seconds to minutes to finish it. The LLM tools block here so their result
says what actually happened (reached altitude, landed, failed) rather than "accepted".
Free of ROS so it can be unit tested with a fake clock.
"""

import time
from typing import Any, Callable, Dict, Optional

Telemetry = Dict[str, Any]
GetTelemetry = Callable[[], Optional[Telemetry]]

# Telemetry arrives at 5 Hz; ignore states older than the command for this long.
STALE_S = 1.0


class FlightWaitError(RuntimeError):
    pass


def wait_for(get: GetTelemetry, done: Callable[[Telemetry], bool], failed: Callable[[Telemetry], Optional[str]],
             timeout: float, what: str, clock=time.monotonic, sleep=time.sleep, poll: float = 0.2) -> Telemetry:
    start = clock()
    while True:
        elapsed = clock() - start
        t = get()
        if t and elapsed >= STALE_S:
            if done(t):
                return t
            reason = failed(t)
            if reason:
                raise FlightWaitError(f"{what} failed: {reason}")
        if elapsed > timeout:
            state = t.get("controller_state") if t else "no telemetry"
            raise FlightWaitError(f"{what} did not finish within {timeout:.0f} s (state {state}).")
        sleep(poll)


def wait_takeoff(get: GetTelemetry, altitude: float, timeout: float, tolerance: float = 0.6, **kw) -> Telemetry:
    return wait_for(
        get,
        done=lambda t: t["controller_state"] == "HOLDING" and abs(t["position"][2] - altitude) <= tolerance,
        failed=lambda t: {"IDLE": "PX4 did not arm or enter offboard (check preflight checks).",
                          "EXTERNAL": f"PX4 left offboard mode ({t['px4_nav_state']})."}.get(t["controller_state"]),
        timeout=timeout, what="Takeoff", **kw)


def wait_landed(get: GetTelemetry, timeout: float, what: str = "Landing", **kw) -> Telemetry:
    return wait_for(
        get,
        done=lambda t: not t["armed"],
        failed=lambda t: None,
        timeout=timeout, what=what, **kw)
