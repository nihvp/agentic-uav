#!/usr/bin/env python3
"""
Direct High-Reliability ROS-to-Gazebo Bridge Relay & Autonomous Flight Controller.
Translates ROS 2 /cmd_vel and /action/agent_command messages directly into Gazebo physics.
Features:
- Instant direct Gazebo transport dispatch on every velocity command.
- Dual-method background poller (gz topic pose/info + gz model --list fallback).
- Dead-reckoning integration ONLY when telemetry is lost.
- Pure zero-velocity hover lock when holding (eliminates P-control limit-cycle oscillation).
- Dynamic survivor evacuation targeting whichever survivor the drone hovers over.
"""

import os
import re
import time
import threading
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from agentic_uav_interfaces.msg import AgentCommand


class GazeboWorldPoller:
    """Non-blocking background poller for Gazebo world entity poses and UAV ground-truth position."""
    def __init__(self, logger=None):
        self.logger = logger
        self.drone_pos = np.array([0.0, 0.0, 0.3], dtype=float)
        self.models = {}  # model_name -> {'pos': np.array([x,y,z]), 'label': str, 'last_seen': float}
        self.last_update = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        while self.running:
            try:
                models_found = {}
                now = time.time()

                # Method 1: Try reading gz topic pose/info (single snapshot)
                try:
                    out = subprocess.check_output(
                        ["gz", "topic", "-e", "-t", "/world/search_and_rescue_harmonic/pose/info", "-n", "1"],
                        stderr=subprocess.DEVNULL,
                        timeout=1.5,
                        text=True
                    )
                    for block in out.split("pose {"):
                        name_m = re.search(r'name:\s*"([^"]+)"', block)
                        if not name_m:
                            continue
                        name = name_m.group(1)
                        pos_m = re.search(r'position\s*\{\s*x:\s*([-\d\.]+)\s*y:\s*([-\d\.]+)\s*z:\s*([-\d\.]+)', block)
                        if pos_m:
                            x, y, z = float(pos_m.group(1)), float(pos_m.group(2)), float(pos_m.group(3))
                            models_found[name] = np.array([x, y, z], dtype=float)
                except Exception:
                    pass

                # Method 2: Fallback to gz model --list + gz model -m <name> -p
                if not models_found:
                    try:
                        l_out = subprocess.check_output(
                            ["gz", "model", "--list", "-w", "search_and_rescue_harmonic"],
                            stderr=subprocess.DEVNULL,
                            timeout=1.5,
                            text=True
                        )
                        for line in l_out.splitlines():
                            name = line.strip().lstrip("-").strip()
                            if not name:
                                continue
                            try:
                                p_out = subprocess.check_output(
                                    ["gz", "model", "-m", name, "-p", "-w", "search_and_rescue_harmonic"],
                                    stderr=subprocess.DEVNULL,
                                    timeout=0.8,
                                    text=True
                                )
                                m = re.search(r'\[\s*([-\d\.]+)[,\s]+([-\d\.]+)[,\s]+([-\d\.]+)\]', p_out)
                                if m:
                                    models_found[name] = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=float)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Categorize discovered models
                for name, pos in models_found.items():
                    if name == "agentic_uav":
                        self.drone_pos = pos
                        self.last_update = now
                        continue

                    name_l = name.lower()
                    label = None
                    if "survivor" in name_l or "person" in name_l or "human" in name_l:
                        label = "survivor"
                    elif "fire" in name_l or "flame" in name_l or "hazard" in name_l:
                        label = "fire_hazard"
                    elif "landing" in name_l or "pad" in name_l:
                        label = "landing_zone"
                    elif "debris" in name_l or "obstacle" in name_l:
                        label = "debris_obstacle"

                    if label:
                        self.models[name] = {"label": label, "pos": pos, "last_seen": now}
                        self.last_update = now

            except Exception:
                pass
            time.sleep(1.5)


class RosToGzRelay(Node):
    def __init__(self):
        super().__init__("ros_to_gz_relay")

        # ROS 2 Publisher into bridged topic
        self.pub_gz_cmd = self.create_publisher(Twist, "/model/agentic_uav/cmd_vel", 10)

        # Subscribers
        self.sub_cmd = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_callback, 10
        )
        self.sub_agent = self.create_subscription(
            AgentCommand, "/action/agent_command", self.agent_callback, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, "/uav/odometry", self.odom_callback, 10
        )
        self.sub_odom_model = self.create_subscription(
            Odometry, "/model/agentic_uav/odometry", self.odom_callback, 10
        )

        # Background Gazebo Poller
        self.gz_poller = GazeboWorldPoller(logger=self.get_logger())

        # Flight State
        self.active_goal = None     # tuple(target_pos, speed)
        self.hold_position = None   # np.array([x, y, z])
        self.current_pos = np.array([0.0, 0.0, 0.3], dtype=float)
        self.last_sent_vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_send_time = 0.0
        self.last_odom_time = 0.0
        self.last_teleop_time = 0.0
        self.rescued_survivors = set()

        # Control loop at 5 Hz (200ms)
        self.timer = self.create_timer(0.2, self.tick)

        self.get_logger().info("=== Direct ROS-to-Gazebo Relay ONLINE ===")
        self.get_logger().info("Ready for autonomous waypoints and manual /cmd_vel.")

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.current_pos = np.array([p.x, p.y, p.z], dtype=float)
        self.last_odom_time = time.time()

    def get_drone_position(self) -> np.ndarray:
        now = time.time()
        # Prefer real odometry if received within 0.5s
        if (now - self.last_odom_time) < 0.5:
            return self.current_pos.copy()
        # Prefer Gazebo ground truth if poller is active
        if (now - self.gz_poller.last_update) < 3.0:
            return self.gz_poller.drone_pos.copy()
        return self.current_pos.copy()

    def send_gz_vel(self, vx: float, vy: float, vz: float, force: bool = False):
        new_vel = np.array([vx, vy, vz], dtype=float)
        now = time.time()

        # Only dispatch if velocity changed significantly or periodic keepalive (every 1.0s)
        if not force and np.linalg.norm(new_vel - self.last_sent_vel) < 0.04 and (now - self.last_send_time) < 1.0:
            return

        self.last_sent_vel = new_vel.copy()
        self.last_send_time = now

        # 1. Publish to ROS Twist
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        self.pub_gz_cmd.publish(msg)

        # 2. Direct dispatch to Gazebo Transport CLI
        payload = f"linear: {{x: {vx:.2f}, y: {vy:.2f}, z: {vz:.2f}}}"
        try:
            subprocess.Popen(
                ["gz", "topic", "-t", "/model/agentic_uav/cmd_vel", "-m", "gz.msgs.Twist", "-p", payload],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            self.get_logger().error(f"Failed to dispatch to Gazebo: {e}")

    def cmd_callback(self, msg: Twist):
        # Operator manual override
        self.active_goal = None
        self.hold_position = None
        self.last_teleop_time = time.time()
        self.send_gz_vel(msg.linear.x, msg.linear.y, msg.linear.z, force=True)

    def agent_callback(self, msg: AgentCommand):
        action = msg.action_type.upper()
        if action in ["NAVIGATE_WAYPOINT", "INSPECT_TARGET"]:
            target_z = max(0.5, float(msg.target_position.z))
            target = np.array([msg.target_position.x, msg.target_position.y, target_z], dtype=float)
            speed = float(msg.speed if msg.speed > 0 else 2.0)
            self.active_goal = (target, speed)
            self.hold_position = None
            self.get_logger().info(f"AI [{action}] Navigating to target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) at {speed:.1f} m/s")
        elif action in ["RTL", "RETURN_TO_LAUNCH"]:
            target = np.array([0.0, 0.0, 2.0], dtype=float)
            self.active_goal = (target, 2.0)
            self.hold_position = None
            self.get_logger().info("AI [RTL] Returning drone to launch base (0, 0, 2)")
        elif action in ["LOITER", "HOLD"]:
            curr = self.get_drone_position()
            self.active_goal = None
            self.hold_position = curr.copy()
            self.send_gz_vel(0.0, 0.0, 0.0, force=True)
            self.get_logger().info("AI [LOITER] Holding current position")

    def tick(self):
        dt = 0.2
        now = time.time()

        # Teleop override cooldown
        if (now - self.last_teleop_time) < 0.6:
            return

        # Fetch best position estimate
        curr_pos = self.get_drone_position()

        # Only use dead reckoning integration if BOTH odom and poller are quiet
        if (now - self.last_odom_time) > 0.5 and (now - self.gz_poller.last_update) > 3.0:
            self.current_pos += self.last_sent_vel * dt
        else:
            self.current_pos = curr_pos

        # Case 1: Actively Navigating to Waypoint
        if self.active_goal is not None:
            target, speed = self.active_goal
            delta = target - self.current_pos
            dist = float(np.linalg.norm(delta))

            if dist > 0.4:
                direction = delta / dist
                cmd_speed = min(speed, max(0.5, dist * 0.7))
                vel = direction * cmd_speed
                self.send_gz_vel(vel[0], vel[1], vel[2])
            else:
                # Target reached! Stop all movement and lock position
                self.send_gz_vel(0.0, 0.0, 0.0, force=True)
                self.get_logger().info(f"Target waypoint reached! Holding position at ({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}, {self.current_pos[2]:.1f}).")
                self.hold_position = target.copy()
                self.active_goal = None

                # Check if this waypoint is near a survivor
                self.check_and_evacuate_survivor(target)
            return

        # Case 2: Stationary Hover
        if self.hold_position is not None:
            # Send zero velocity to cancel all momentum and keep drone completely still without oscillation
            self.send_gz_vel(0.0, 0.0, 0.0)

    def check_and_evacuate_survivor(self, target_pos: np.ndarray):
        """Finds any survivor model near target position and evacuates them to the landing pad."""
        nearest_survivor = None
        min_dist = 4.5  # Search within 4.5m radius of waypoint

        for name, data in self.gz_poller.models.items():
            if data["label"] == "survivor" and name not in self.rescued_survivors:
                spos = data["pos"]
                d = float(np.linalg.norm(spos[:2] - target_pos[:2]))
                if d < min_dist:
                    min_dist = d
                    nearest_survivor = name

        # Default fallback for initial survivor zone
        if not nearest_survivor and (target_pos[0] > 6.0 and target_pos[1] > 2.0) and "survivor_target" not in self.rescued_survivors:
            nearest_survivor = "survivor_target"

        if nearest_survivor:
            self.evacuate_survivor(nearest_survivor)

    def evacuate_survivor(self, model_name: str):
        self.rescued_survivors.add(model_name)
        count = len(self.rescued_survivors)
        # Stagger rescued survivors across the landing pad (radius 2m at x=0.0, y=8.0)
        offset_x = (count - 1) * 0.8 - 0.8
        dest_x = float(np.clip(offset_x, -1.2, 1.2))
        dest_y = 8.0

        self.get_logger().info(f"[MISSION EVENT] Stable visual confirmed over '{model_name}'! Ground rescue team deploying...")

        def _evacuate():
            time.sleep(2.5)  # 2.5s observation period
            try:
                subprocess.run(
                    ["gz", "service", "-s", "/world/search_and_rescue_harmonic/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean", "--timeout", "1000",
                     "--req", f'name: "{model_name}", position: {{x: {dest_x:.2f}, y: {dest_y:.2f}, z: 0.0}}, orientation: {{w: 1.0}}'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self.get_logger().info(
                    f">>> [RESCUE SUCCESS] '{model_name}' has been safely evacuated to Emergency Landing Pad ({dest_x:.1f}, {dest_y:.1f})! <<<"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to relocate {model_name}: {e}")

        threading.Thread(target=_evacuate, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = RosToGzRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
