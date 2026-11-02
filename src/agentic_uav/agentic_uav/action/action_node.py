"""Action layer: the single owner of PX4 offboard control.

Exposes
  /action/navigate_to     agentic_uav_interfaces/action/NavigateTo
  /action/flight_command  agentic_uav_interfaces/srv/FlightCommand
  /action/telemetry       agentic_uav_interfaces/msg/FlightTelemetry
and talks to PX4 over uXRCE-DDS (px4_msgs).
"""

import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import (
    BatteryStatus,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

from agentic_uav_interfaces.action import NavigateTo
from agentic_uav_interfaces.msg import FlightTelemetry
from agentic_uav_interfaces.srv import FlightCommand
from agentic_uav.action.offboard import ControllerConfig, OffboardController
from agentic_uav.common.frames import ned_to_enu, yaw_ned_to_enu

NAV_STATE_NAMES = {
    v: k[len("NAVIGATION_STATE_"):]
    for k, v in vars(VehicleStatus).items()
    if k.startswith("NAVIGATION_STATE_") and isinstance(v, int)
}

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class ActionNode(Node):
    def __init__(self):
        super().__init__("action_node")

        cfg = ControllerConfig(
            takeoff_altitude=self.declare_parameter("takeoff_altitude", 4.0).value,
            acceptance_radius=self.declare_parameter("acceptance_radius", 0.5).value,
            geofence_radius=self.declare_parameter("geofence_radius", 150.0).value,
            geofence_max_alt=self.declare_parameter("geofence_max_alt", 25.0).value,
            min_nav_altitude=self.declare_parameter("min_nav_altitude", 1.0).value,
            rate_hz=self.declare_parameter("rate_hz", 20.0).value,
        )
        self.nav_timeout_s = self.declare_parameter("nav_timeout_s", 120.0).value
        # PX4 versions topic names (e.g. vehicle_status_v1); confirm with `ros2 topic list`.
        topic_status = self.declare_parameter("px4.status_topic", "/fmu/out/vehicle_status_v1").value
        topic_lpos = self.declare_parameter("px4.local_position_topic", "/fmu/out/vehicle_local_position").value
        topic_batt = self.declare_parameter("px4.battery_topic", "/fmu/out/battery_status").value

        self.ctrl = OffboardController(cfg)
        self.battery = math.nan
        self.vel_enu = (0.0, 0.0, 0.0)
        cb = ReentrantCallbackGroup()

        self.create_subscription(VehicleStatus, topic_status, self._on_status, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, topic_lpos, self._on_local_position, PX4_QOS)
        self.create_subscription(BatteryStatus, topic_batt, self._on_battery, PX4_QOS)

        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.pub_telem = self.create_publisher(FlightTelemetry, "/action/telemetry", 10)

        self.create_service(FlightCommand, "/action/flight_command", self._on_flight_command, callback_group=cb)
        self.nav_server = ActionServer(
            self, NavigateTo, "/action/navigate_to",
            execute_callback=self._execute_nav,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=cb,
        )

        self.create_timer(1.0 / cfg.rate_hz, self._tick)
        self.create_timer(0.2, self._publish_telemetry)
        self.get_logger().info(f"Action layer up (status topic: {topic_status}).")

    # ---- PX4 inputs -----------------------------------------------------

    def _on_status(self, msg: VehicleStatus):
        self.ctrl.update_status(msg.arming_state, msg.nav_state)

    def _on_local_position(self, msg: VehicleLocalPosition):
        self.ctrl.update_position((msg.x, msg.y, msg.z), msg.heading, msg.xy_valid and msg.z_valid)
        self.vel_enu = ned_to_enu((msg.vx, msg.vy, msg.vz))

    def _on_battery(self, msg: BatteryStatus):
        self.battery = float(msg.remaining) * 100.0

    # ---- control loop ---------------------------------------------------

    def _now_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def _tick(self):
        prev = self.ctrl.state
        out = self.ctrl.tick()
        if self.ctrl.state != prev:
            self.get_logger().info(f"{prev.value} -> {self.ctrl.state.value} {self.ctrl.last_error}".rstrip())

        stamp = self._now_us()
        if out.stream_offboard and out.setpoint_ned is not None:
            mode = OffboardControlMode()
            mode.position = True
            mode.timestamp = stamp
            self.pub_mode.publish(mode)

            sp = TrajectorySetpoint()
            sp.position = [float(v) for v in out.setpoint_ned]
            sp.velocity = [math.nan] * 3
            sp.acceleration = [math.nan] * 3
            sp.jerk = [math.nan] * 3
            sp.yaw = float(out.yaw_ned)
            sp.yawspeed = math.nan
            sp.timestamp = stamp
            self.pub_sp.publish(sp)

        for req in out.commands:
            cmd = VehicleCommand()
            cmd.command = req.command
            for name, value in req.params.items():
                setattr(cmd, name, float(value))
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            cmd.timestamp = stamp
            self.pub_cmd.publish(cmd)

    # ---- interfaces -----------------------------------------------------

    def _on_flight_command(self, request, response):
        command = request.command.strip().upper()
        handlers = {
            "TAKEOFF": lambda: self.ctrl.takeoff(request.altitude),
            "HOLD": self.ctrl.hold,
            "LAND": self.ctrl.land,
            "RTL": self.ctrl.rtl,
        }
        if command not in handlers:
            response.success, response.message = False, f"Unknown command '{request.command}'."
        else:
            response.success, response.message = handlers[command]()
        self.get_logger().info(f"FlightCommand {command}: {response.message}")
        return response

    def _execute_nav(self, goal_handle):
        g = goal_handle.request
        target = (g.target.x, g.target.y, g.target.z)
        ok, message, seq = self.ctrl.go_to(target, g.yaw, g.acceptance_radius)
        result = NavigateTo.Result()
        if not ok:
            goal_handle.abort()
            result.success, result.message = False, message
            return result

        self.get_logger().info(f"NavigateTo {target}: accepted")
        deadline = time.monotonic() + self.nav_timeout_s
        feedback = NavigateTo.Feedback()
        while rclpy.ok():
            outcome = self.ctrl.goal_result(seq)
            if outcome is not None:
                break
            if goal_handle.is_cancel_requested:
                self.ctrl.hold()
                goal_handle.canceled()
                result.success, result.message = False, "Cancelled; holding position."
                return result
            if time.monotonic() > deadline:
                self.ctrl.hold()
                outcome = (False, f"Timed out after {self.nav_timeout_s:.0f} s; holding position.")
                break
            pos = self.ctrl.position_enu
            if pos is not None:
                feedback.current_position.x, feedback.current_position.y, feedback.current_position.z = pos
            feedback.distance_remaining = float(self.ctrl.distance_to_target())
            goal_handle.publish_feedback(feedback)
            time.sleep(0.2)

        result.success, result.message = outcome
        pos = self.ctrl.position_enu
        if pos is not None:
            result.final_position.x, result.final_position.y, result.final_position.z = pos
        goal_handle.succeed() if result.success else goal_handle.abort()
        return result

    def _publish_telemetry(self):
        msg = FlightTelemetry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.armed = self.ctrl.armed
        msg.px4_nav_state = NAV_STATE_NAMES.get(self.ctrl.nav_state, str(self.ctrl.nav_state))
        msg.controller_state = self.ctrl.state.value
        pos = self.ctrl.position_enu
        if pos is not None:
            msg.current_pose.position.x, msg.current_pose.position.y, msg.current_pose.position.z = pos
            yaw = yaw_ned_to_enu(self.ctrl.heading_ned) if math.isfinite(self.ctrl.heading_ned) else 0.0
            msg.current_pose.orientation.z = math.sin(yaw / 2.0)
            msg.current_pose.orientation.w = math.cos(yaw / 2.0)
        msg.current_twist.linear.x, msg.current_twist.linear.y, msg.current_twist.linear.z = self.vel_enu
        msg.battery_percentage = self.battery
        msg.geofence_ok = self.ctrl.geofence_ok()
        msg.is_in_air = self.ctrl.in_air()
        self.pub_telem.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
