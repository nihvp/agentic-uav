#!/usr/bin/env python3
"""
Action Layer Node.
Translates symbolic agent commands into physical flight instructions for Gazebo Harmonic
and dispatches atomic digital/physical tool actuations.
Streams high-frequency offboard setpoints and publishes flight telemetry.
"""

import json
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from agentic_uav_interfaces.msg import AgentCommand, FlightTelemetry
from agentic_uav.action.px4_controller import PX4Controller, FlightState
from agentic_uav.action.tool_actuators import ToolActuators


class ActionNode(Node):
    def __init__(self):
        super().__init__("action_node")

        # Declare parameters
        self.declare_parameter("takeoff_altitude", 4.0)
        self.declare_parameter("max_flight_speed", 4.0)
        self.declare_parameter("geofence_radius", 150.0)
        self.declare_parameter("geofence_max_alt", 25.0)
        self.declare_parameter("stream_rate_hz", 20.0)

        takeoff_alt = self.get_parameter("takeoff_altitude").get_parameter_value().double_value
        max_speed = self.get_parameter("max_flight_speed").get_parameter_value().double_value
        geo_radius = self.get_parameter("geofence_radius").get_parameter_value().double_value
        geo_alt = self.get_parameter("geofence_max_alt").get_parameter_value().double_value

        self.px4 = PX4Controller(
            takeoff_altitude=takeoff_alt,
            max_speed=max_speed,
            geofence_radius=geo_radius,
            geofence_max_alt=geo_alt
        )
        self.actuators = ToolActuators()

        # Flight state tracking
        self.sim_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.sim_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.has_odom = False
        self.active_navigating = True

        # Subscribers
        self.sub_cmd = self.create_subscription(
            AgentCommand, "/action/agent_command", self.command_callback, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, "/uav/odometry", self.odometry_callback, 10
        )
        # Manual velocity teleop subscriber: forwards /cmd_vel directly to /model/agentic_uav/cmd_vel
        self.sub_manual = self.create_subscription(
            Twist, "/cmd_vel", self.manual_cmd_callback, 10
        )

        # Publishers
        self.pub_telem = self.create_publisher(FlightTelemetry, "/action/flight_telemetry", 10)
        self.pub_setpoint = self.create_publisher(PoseStamped, "/fmu/in/trajectory_setpoint", 10)
        self.pub_mavros_setpoint = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        # Publish to the EXACT topic that VelocityControl listens on.
        # The default VelocityControl topic for model "agentic_uav" is /model/agentic_uav/cmd_vel.
        # The bridge maps this same name on both ROS and GZ sides.
        self.pub_cmd_vel = self.create_publisher(Twist, "/model/agentic_uav/cmd_vel", 10)

        # Automatically arm and takeoff for autonomous mission start
        self.px4.arm_and_takeoff()

        # High-frequency flight control timer (20 Hz)
        rate = self.get_parameter("stream_rate_hz").get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info("Action Layer active: publishing to /model/agentic_uav/cmd_vel at 20 Hz.")

    def command_callback(self, msg: AgentCommand):
        self.get_logger().info(f"Received AgentCommand [{msg.command_id}] Type: {msg.action_type}")

        action = msg.action_type.upper()

        if action == "NAVIGATE_WAYPOINT":
            self.active_navigating = True
            self.px4.set_nav_waypoint(
                msg.target_position.x,
                msg.target_position.y,
                msg.target_position.z,
                speed=msg.speed if msg.speed > 0 else 2.0,
                yaw=msg.target_yaw if msg.target_yaw != 0 else None
            )
            self.get_logger().info(
                f"NAVIGATE_WAYPOINT -> Target: ({msg.target_position.x:.1f}, {msg.target_position.y:.1f}, {msg.target_position.z:.1f}) Speed: {msg.speed} m/s"
            )

        elif action == "INSPECT_TARGET":
            self.active_navigating = True
            self.px4.set_nav_waypoint(
                msg.target_position.x,
                msg.target_position.y,
                msg.target_position.z,
                speed=1.5
            )
            # Execute inspection tool actuator (camera gimbal track)
            if msg.tool_name:
                try:
                    params = json.loads(msg.tool_parameters) if msg.tool_parameters else {}
                except Exception:
                    params = {}
                self.actuators.execute_tool(msg.tool_name, params)

        elif action == "LOITER":
            self.active_navigating = False
            self.px4.set_loiter(duration=msg.loiter_duration if msg.loiter_duration > 0 else 5.0)

        elif action in ["RTL", "RETURN_TO_LAUNCH"]:
            self.active_navigating = True
            self.px4.trigger_rtl()

        # Handle atomic digital tool invocations
        if msg.tool_name and action not in ["INSPECT_TARGET"]:
            try:
                params = json.loads(msg.tool_parameters) if msg.tool_parameters else {}
            except Exception:
                params = {}
            success, msg_str = self.actuators.execute_tool(msg.tool_name, params)
            self.get_logger().info(f"Tool invocation '{msg.tool_name}': Success={success} ({msg_str})")

    def manual_cmd_callback(self, msg: Twist):
        # Forward operator manual teleop commands (/cmd_vel) directly to Gazebo (/model/agentic_uav/cmd_vel)
        self.pub_cmd_vel.publish(msg)
        # If manual operator sends a movement command, yield autonomous navigation authority
        if abs(msg.linear.x) > 0.05 or abs(msg.linear.y) > 0.05 or abs(msg.linear.z) > 0.05:
            self.active_navigating = False

    def odometry_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.sim_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.sim_vel = np.array([v.x, v.y, v.z], dtype=np.float32)
        self.px4.update_odometry(self.sim_pos, self.sim_vel)
        if not self.has_odom:
            self.get_logger().info("First odometry message received from Gazebo!")
        self.has_odom = True

    def control_loop(self):
        dt = 0.05
        # Compute setpoint based on current position feedback
        target_pos, des_vel, target_yaw = self.px4.compute_step(dt=dt)

        if not self.has_odom:
            # No odometry yet — still publish velocity so the drone starts moving.
            # Without odom feedback, the controller thinks it's at (0,0,0) and
            # will keep commanding velocity toward the takeoff target.
            self.get_logger().info(
                "No odometry yet — commanding velocity anyway.",
                throttle_duration_sec=5.0
            )

        # Publish PX4 / MAVROS trajectory setpoint
        sp_msg = PoseStamped()
        sp_msg.header.stamp = self.get_clock().now().to_msg()
        sp_msg.header.frame_id = "map"
        sp_msg.pose.position.x = float(target_pos[0])
        sp_msg.pose.position.y = float(target_pos[1])
        sp_msg.pose.position.z = float(target_pos[2])

        # Yaw to quaternion
        cy = float(np.cos(target_yaw * 0.5))
        sy = float(np.sin(target_yaw * 0.5))
        sp_msg.pose.orientation.w = cy
        sp_msg.pose.orientation.z = sy

        self.pub_setpoint.publish(sp_msg)
        self.pub_mavros_setpoint.publish(sp_msg)

        # Publish Twist velocity to move the drone in Gazebo
        vel_mag = float(np.linalg.norm(des_vel))
        if vel_mag > 0.02 or self.active_navigating:
            cmd_vel = Twist()
            cmd_vel.linear.x = float(des_vel[0])
            cmd_vel.linear.y = float(des_vel[1])
            cmd_vel.linear.z = float(des_vel[2])
            self.pub_cmd_vel.publish(cmd_vel)
            if vel_mag <= 0.02:
                if self.active_navigating:
                    self.get_logger().info(
                        f"Target waypoint reached! Holding at ({self.sim_pos[0]:.1f}, {self.sim_pos[1]:.1f}, {self.sim_pos[2]:.1f})."
                    )
                self.active_navigating = False

        # Stream Flight Telemetry to Reasoning Layer
        telem = FlightTelemetry()
        telem.header.stamp = self.get_clock().now().to_msg()
        telem.header.frame_id = "map"
        telem.armed = bool(self.px4.armed)
        telem.flight_mode = str(self.px4.state.name)
        telem.battery_percentage = float(self.px4.battery_level)
        telem.geofence_ok = bool(self.px4.check_geofence())
        telem.is_in_air = bool(float(self.sim_pos[2]) > 0.3)

        telem.current_pose.position.x = float(self.sim_pos[0])
        telem.current_pose.position.y = float(self.sim_pos[1])
        telem.current_pose.position.z = float(self.sim_pos[2])
        telem.current_twist.linear.x = float(des_vel[0])
        telem.current_twist.linear.y = float(des_vel[1])
        telem.current_twist.linear.z = float(des_vel[2])

        self.pub_telem.publish(telem)


def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
