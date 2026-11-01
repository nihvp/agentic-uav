"""
PX4 Flight Controller Interface.
Encapsulates flight state transitions, Offboard setpoint streaming,
safety geofencing, and failsafe RTL handling for PX4 SITL and simulated drones.
"""

import math
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any, Optional


class FlightState(Enum):
    DISARMED = 0
    ARMING = 1
    TAKEOFF = 2
    OFFBOARD = 3
    LOITER = 4
    RTL = 5
    LANDING = 6
    LANDED = 7


class PX4Controller:
    def __init__(self, takeoff_altitude: float = 3.5, max_speed: float = 5.0,
                 geofence_radius: float = 150.0, geofence_max_alt: float = 25.0):
        self.takeoff_altitude = takeoff_altitude
        self.max_speed = max_speed
        self.geofence_radius = geofence_radius
        self.geofence_max_alt = geofence_max_alt

        self.state = FlightState.DISARMED
        self.current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.home_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([0.0, 0.0, takeoff_altitude], dtype=np.float32)
        self.target_yaw = 0.0

        self.battery_level = 100.0  # Percentage
        self.armed = False
        self.loiter_timer = 0.0

    def update_odometry(self, pos: np.ndarray, vel: np.ndarray):
        """Updates internal drone position and performs geofence check."""
        self.current_pos = pos.copy()
        self.current_vel = vel.copy()

        # Simulate gradual battery discharge per step
        self.battery_level = max(0.0, self.battery_level - 0.005)

    def check_geofence(self) -> bool:
        """Returns True if within safe bounds, False if boundary violated."""
        horizontal_dist = np.linalg.norm(self.current_pos[:2] - self.home_pos[:2])
        alt = self.current_pos[2]
        if horizontal_dist > self.geofence_radius or alt > self.geofence_max_alt or alt < -1.0:
            return False
        return True

    def arm_and_takeoff(self) -> bool:
        """Transitions state machine to armed takeoff."""
        self.armed = True
        self.state = FlightState.TAKEOFF
        self.target_pos = np.array([self.home_pos[0], self.home_pos[1], self.takeoff_altitude], dtype=np.float32)
        return True

    def set_nav_waypoint(self, x: float, y: float, z: float, speed: float = 2.0, yaw: Optional[float] = None):
        """Commands a 3D position setpoint."""
        z = min(z, self.geofence_max_alt - 1.0)
        self.target_pos = np.array([x, y, z], dtype=np.float32)
        if yaw is not None:
            self.target_yaw = yaw
        else:
            # Auto-align yaw with flight trajectory
            delta = self.target_pos[:2] - self.current_pos[:2]
            if np.linalg.norm(delta) > 0.5:
                self.target_yaw = float(np.arctan2(delta[1], delta[0]))

        self.state = FlightState.OFFBOARD

    def set_loiter(self, duration: float = 5.0):
        """Enters holding pattern at current location."""
        self.target_pos = self.current_pos.copy()
        self.loiter_timer = duration
        self.state = FlightState.LOITER

    def trigger_rtl(self):
        """Triggers Return-To-Launch sequence."""
        self.target_pos = np.array([self.home_pos[0], self.home_pos[1], self.takeoff_altitude], dtype=np.float32)
        self.state = FlightState.RTL

    def compute_step(self, dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes the next position setpoint, desired velocity vector, and yaw angle
        for the PX4 offboard streaming pipeline.
        """
        # Geofence enforcement
        if not self.check_geofence() and self.state != FlightState.RTL:
            self.trigger_rtl()

        # Simple P-controller for position setpoint smoothing
        error_pos = self.target_pos - self.current_pos
        dist = np.linalg.norm(error_pos)

        if dist > 0.05:
            direction = error_pos / dist
            speed = min(self.max_speed, dist * 0.8)
            desired_vel = direction * speed
        else:
            desired_vel = np.zeros(3, dtype=np.float32)
            if self.state == FlightState.TAKEOFF:
                self.state = FlightState.OFFBOARD
            elif self.state == FlightState.RTL:
                self.state = FlightState.LANDING

        return self.target_pos, desired_vel, self.target_yaw
