"""PX4 offboard flight state machine, free of ROS so it can be unit tested.

action_node feeds it PX4 state (local position, arming and navigation state) and
calls tick() at a fixed rate. tick() returns what to publish: whether to stream
OffboardControlMode, the TrajectorySetpoint (NED) and any VehicleCommands.

This is the only component that decides PX4 setpoints. PX4 does the actual flight
control; we never command velocities to the simulator directly.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from agentic_uav.common.frames import enu_to_ned, ned_to_enu, yaw_enu_to_ned

# Mirrors px4_msgs VehicleCommand / VehicleStatus constants (kept here so this
# module does not need px4_msgs).
CMD_DO_SET_MODE = 176
CMD_COMPONENT_ARM_DISARM = 400
CMD_NAV_LAND = 21
CMD_NAV_RETURN_TO_LAUNCH = 20
ARMING_STATE_ARMED = 2
NAV_STATE_OFFBOARD = 14
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0


class State(Enum):
    IDLE = "IDLE"                # on the ground / disarmed
    STARTING = "STARTING"        # streaming setpoints before offboard + arm
    TAKING_OFF = "TAKING_OFF"
    HOLDING = "HOLDING"
    NAVIGATING = "NAVIGATING"
    HANDED_OFF = "HANDED_OFF"    # LAND / RTL delegated to PX4 auto modes
    EXTERNAL = "EXTERNAL"        # PX4 left offboard (failsafe or pilot)


@dataclass
class ControllerConfig:
    takeoff_altitude: float = 4.0
    acceptance_radius: float = 0.5
    geofence_radius: float = 150.0
    geofence_max_alt: float = 25.0
    min_nav_altitude: float = 1.0
    rate_hz: float = 20.0
    # Setpoints PX4 must receive before it accepts offboard mode.
    preroll_ticks: int = 10
    # How long to wait for PX4 to confirm offboard + armed after requesting it.
    start_timeout_s: float = 5.0


@dataclass
class VehicleCommandRequest:
    command: int
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class TickOutput:
    stream_offboard: bool = False
    setpoint_ned: Optional[Tuple[float, float, float]] = None
    yaw_ned: float = math.nan
    commands: List[VehicleCommandRequest] = field(default_factory=list)


class OffboardController:
    def __init__(self, config: ControllerConfig):
        self.cfg = config
        self.state = State.IDLE
        self.last_error = ""

        self.pos_ned: Optional[Tuple[float, float, float]] = None
        self.heading_ned = math.nan
        self.armed = False
        self.nav_state = -1

        self._target_ned: Optional[Tuple[float, float, float]] = None
        self._target_yaw_ned = math.nan
        self._radius = config.acceptance_radius
        self._ticks_in_state = 0
        self._offboard_confirmed = False

        self._goal_seq = 0
        self._goal_results: Dict[int, Tuple[bool, str]] = {}
        self._queued_commands: List[VehicleCommandRequest] = []

    # ---- PX4 inputs -----------------------------------------------------

    def update_position(self, pos_ned: Tuple[float, float, float], heading_ned: float, valid: bool):
        if valid:
            self.pos_ned = tuple(float(v) for v in pos_ned)
            self.heading_ned = float(heading_ned)

    def update_status(self, arming_state: int, nav_state: int):
        self.armed = arming_state == ARMING_STATE_ARMED
        self.nav_state = nav_state

    # ---- derived state --------------------------------------------------

    @property
    def position_enu(self) -> Optional[Tuple[float, float, float]]:
        return ned_to_enu(self.pos_ned) if self.pos_ned is not None else None

    def in_air(self) -> bool:
        return self.armed and self.pos_ned is not None and -self.pos_ned[2] > 0.3

    def geofence_ok(self, enu: Optional[Tuple[float, float, float]] = None) -> bool:
        p = enu if enu is not None else self.position_enu
        if p is None:
            return True
        return math.hypot(p[0], p[1]) <= self.cfg.geofence_radius and p[2] <= self.cfg.geofence_max_alt

    def distance_to_target(self) -> float:
        if self._target_ned is None or self.pos_ned is None:
            return math.nan
        return math.dist(self._target_ned, self.pos_ned)

    # ---- commands -------------------------------------------------------

    def takeoff(self, altitude: Optional[float] = None) -> Tuple[bool, str]:
        if self.pos_ned is None:
            return False, "No valid PX4 local position yet."
        alt = altitude if altitude and altitude > 0 else self.cfg.takeoff_altitude
        if alt > self.cfg.geofence_max_alt:
            return False, f"Takeoff altitude {alt:.1f} m exceeds geofence ceiling {self.cfg.geofence_max_alt:.1f} m."
        self._abort_goal("Preempted by takeoff.")
        self._target_ned = (self.pos_ned[0], self.pos_ned[1], -alt)
        self._target_yaw_ned = self.heading_ned
        self._radius = self.cfg.acceptance_radius
        self._enter(State.STARTING)
        return True, f"Taking off to {alt:.1f} m."

    def go_to(self, target_enu: Tuple[float, float, float], yaw_enu: float = math.nan,
              radius: float = 0.0) -> Tuple[bool, str, int]:
        if self.state in (State.STARTING, State.TAKING_OFF):
            return False, "Still taking off; wait for controller_state HOLDING, then navigate.", -1
        if self.state not in (State.HOLDING, State.NAVIGATING):
            return False, f"Cannot navigate in state {self.state.value}; take off first.", -1
        if target_enu[2] < self.cfg.min_nav_altitude:
            return False, f"Target altitude {target_enu[2]:.1f} m below minimum {self.cfg.min_nav_altitude:.1f} m.", -1
        if not self.geofence_ok(target_enu):
            return False, "Target is outside the geofence.", -1
        self._abort_goal("Preempted by a new navigation goal.")
        self._goal_seq += 1
        self._target_ned = enu_to_ned(target_enu)
        self._target_yaw_ned = self.heading_ned if math.isnan(yaw_enu) else yaw_enu_to_ned(yaw_enu)
        self._radius = radius if radius > 0 else self.cfg.acceptance_radius
        self._enter(State.NAVIGATING)
        return True, "Navigating.", self._goal_seq

    def goal_result(self, seq: int) -> Optional[Tuple[bool, str]]:
        """None while the goal is active, else (success, message)."""
        return self._goal_results.get(seq)

    def hold(self) -> Tuple[bool, str]:
        if self.state not in (State.HOLDING, State.NAVIGATING, State.TAKING_OFF):
            return False, f"Cannot hold in state {self.state.value}."
        self._abort_goal("Cancelled by hold.")
        self._hold_here()
        return True, "Holding position."

    def land(self) -> Tuple[bool, str]:
        return self._hand_off(CMD_NAV_LAND, "Landing.")

    def rtl(self) -> Tuple[bool, str]:
        return self._hand_off(CMD_NAV_RETURN_TO_LAUNCH, "Returning to launch.")

    # ---- loop -----------------------------------------------------------

    def tick(self) -> TickOutput:
        self._ticks_in_state += 1
        out = TickOutput()
        out.commands, self._queued_commands = self._queued_commands, []

        if self.state in (State.IDLE, State.HANDED_OFF, State.EXTERNAL):
            if not self.armed and self.state != State.IDLE:
                self._enter(State.IDLE)
            return out

        if self.armed and self.nav_state == NAV_STATE_OFFBOARD:
            self._offboard_confirmed = True
        elif self._offboard_confirmed:
            self._abort_goal("PX4 left offboard mode (failsafe or pilot takeover).")
            self.last_error = "PX4 left offboard mode."
            self._enter(State.EXTERNAL)
            return out

        out.stream_offboard = True
        out.setpoint_ned = self._target_ned
        out.yaw_ned = self._target_yaw_ned

        if self.state == State.STARTING:
            if self._ticks_in_state == self.cfg.preroll_ticks:
                out.commands.append(VehicleCommandRequest(
                    CMD_DO_SET_MODE, {"param1": 1.0, "param2": PX4_CUSTOM_MAIN_MODE_OFFBOARD}))
                out.commands.append(VehicleCommandRequest(CMD_COMPONENT_ARM_DISARM, {"param1": 1.0}))
            elif self._offboard_confirmed:
                self._enter(State.TAKING_OFF)
            elif self._ticks_in_state > self.cfg.preroll_ticks + self.cfg.start_timeout_s * self.cfg.rate_hz:
                self.last_error = "PX4 did not accept offboard mode / arming (check preflight checks)."
                self._enter(State.IDLE)
                out.stream_offboard = False
            return out

        if self.state == State.TAKING_OFF and self._reached():
            self._enter(State.HOLDING)
        elif self.state == State.NAVIGATING and self._reached():
            self._goal_results[self._goal_seq] = (True, "Reached target.")
            self._enter(State.HOLDING)
        return out

    # ---- internals ------------------------------------------------------

    def _reached(self) -> bool:
        d = self.distance_to_target()
        return not math.isnan(d) and d <= self._radius

    def _hold_here(self):
        if self.pos_ned is not None:
            self._target_ned = self.pos_ned
        self._enter(State.HOLDING)

    def _hand_off(self, command: int, message: str) -> Tuple[bool, str]:
        if not self.armed:
            return False, "Vehicle is not armed."
        self._abort_goal(f"Cancelled: {message}")
        self._queued_commands.append(VehicleCommandRequest(command))
        self._enter(State.HANDED_OFF)
        return True, message

    def _abort_goal(self, reason: str):
        if self.state == State.NAVIGATING and self._goal_seq not in self._goal_results:
            self._goal_results[self._goal_seq] = (False, reason)

    def _enter(self, state: State):
        if state in (State.IDLE, State.HANDED_OFF, State.EXTERNAL):
            self._offboard_confirmed = False
        self.state = state
        self._ticks_in_state = 0
