"""Capture one experiment scene: fly to its viewpoint, hover steady, save frames.

Runs in the agent container with the stack up (action layer and camera bridge):
  python3 -m agentic_uav.experiment.capture --scene 3 [--frames 3] [--rtl]

Uses the action layer's normal interfaces (takeoff, navigate_to) and the same sensor
topics as the perception layer. From the scene file it reads only the viewpoint and
heading (where to fly), never the label or the people. Takes off first if the vehicle
is on the ground; stays airborne afterwards unless --rtl, so scenes can follow one
another. scripts/capture_scenes.sh runs spawn + capture for every scene.
"""

import argparse
import math
import sys
import threading
import time
from collections import deque

import rclpy
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

from agentic_uav_interfaces.action import NavigateTo
from agentic_uav_interfaces.msg import FlightTelemetry
from agentic_uav_interfaces.srv import FlightCommand
from agentic_uav.common.frames import ned_to_enu, yaw_ned_to_enu
from agentic_uav.experiment.capture_io import (horizontal_offset, is_settled, round_list, save_frame,
                                               scene_dir, write_meta)
from agentic_uav.experiment.scenes import load
from agentic_uav.perception.camera_model import camera_pose_world
from agentic_uav.reasoning.flight_waits import wait_landed, wait_takeoff

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def wait_future(future, timeout: float):
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    if not done.wait(timeout):
        raise TimeoutError(f"No response within {timeout:.0f} s.")
    return future.result()


def camera_info_dict(msg: CameraInfo):
    return {"width": msg.width, "height": msg.height, "k": round_list(msg.k, 3), "frame_id": msg.header.frame_id}


class CaptureNode(Node):
    def __init__(self, mount_xyz, mount_pitch):
        super().__init__("capture_node")
        self.mount_xyz, self.mount_pitch = list(mount_xyz), float(mount_pitch)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.telemetry = None
        self.depth = None
        self.rgb_info = self.depth_info = None
        self.pos_ned = None
        self.heading_ned = math.nan
        self.q = None
        self.motion = deque(maxlen=200)      # (t, speed, heading) from PX4 local position
        self.wanted = 0
        self.min_gap_s = 0.5
        self.frames = []                     # (wall time, bgr, depth, pos_ned, q, heading_ned)
        cb = ReentrantCallbackGroup()
        self.create_subscription(FlightTelemetry, "/action/telemetry", self._on_telemetry, 10)
        self.create_subscription(Image, "/camera/image", self._on_rgb, 1)
        self.create_subscription(Image, "/camera/depth", self._on_depth, 1)
        self.create_subscription(CameraInfo, "/camera/camera_info", lambda m: setattr(self, "rgb_info", m), 1)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", lambda m: setattr(self, "depth_info", m), 1)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._on_att, PX4_QOS)
        self.nav = ActionClient(self, NavigateTo, "/action/navigate_to", callback_group=cb)
        self.flight = self.create_client(FlightCommand, "/action/flight_command", callback_group=cb)

    # ---- inputs ---------------------------------------------------------

    def _on_telemetry(self, msg: FlightTelemetry):
        p = msg.current_pose.position
        self.telemetry = {"armed": msg.armed, "px4_nav_state": msg.px4_nav_state,
                          "controller_state": msg.controller_state, "in_air": msg.is_in_air,
                          "position": [p.x, p.y, p.z], "last_event": msg.last_event}

    def _on_lpos(self, msg: VehicleLocalPosition):
        if not (msg.xy_valid and msg.z_valid):
            return
        self.pos_ned = (msg.x, msg.y, msg.z)
        self.heading_ned = msg.heading
        self.motion.append((time.monotonic(), math.sqrt(msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2), msg.heading))

    def _on_att(self, msg: VehicleAttitude):
        self.q = tuple(float(v) for v in msg.q)

    def _on_depth(self, msg: Image):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _on_rgb(self, msg: Image):
        with self.lock:
            if len(self.frames) >= self.wanted or self.depth is None or self.pos_ned is None or self.q is None:
                return
            now = time.monotonic()
            if self.frames and now - self.frames[-1][0] < self.min_gap_s:
                return
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frames.append((now, bgr, self.depth.copy(), self.pos_ned, self.q, self.heading_ned))

    # ---- actions --------------------------------------------------------

    def get_telemetry(self):
        return dict(self.telemetry) if self.telemetry else None

    def command(self, name: str, altitude: float = 0.0):
        if not self.flight.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Action layer flight_command service unavailable (is the stack running?).")
        res = wait_future(self.flight.call_async(FlightCommand.Request(command=name, altitude=float(altitude))), 10.0)
        if not res.success:
            raise RuntimeError(f"{name}: {res.message}")

    def navigate(self, x, y, z, yaw_rad, timeout_s: float):
        if not self.nav.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action layer navigate_to server unavailable.")
        goal = NavigateTo.Goal()
        goal.target.x, goal.target.y, goal.target.z = float(x), float(y), float(z)
        goal.yaw, goal.acceptance_radius = float(yaw_rad), 0.3
        handle = wait_future(self.nav.send_goal_async(goal), 10.0)
        if not handle.accepted:
            raise RuntimeError("Navigation goal rejected.")
        res = wait_future(handle.get_result_async(), timeout_s).result
        if not res.success:
            raise RuntimeError(f"Navigation failed: {res.message}")

    def wait_settled(self, timeout_s: float) -> float:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if is_settled(list(self.motion), time.monotonic()):
                return time.monotonic() - start
            time.sleep(0.2)
        raise RuntimeError(f"Vehicle did not settle within {timeout_s:.0f} s.")

    def grab(self, n: int, timeout_s: float):
        with self.lock:
            self.frames, self.wanted = [], n
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            with self.lock:
                if len(self.frames) >= n:
                    self.wanted = 0
                    return list(self.frames)
            time.sleep(0.05)
        raise RuntimeError(f"Got {len(self.frames)}/{n} frames in {timeout_s:.0f} s "
                           "(camera bridge, depth and PX4 pose all needed).")


def capture_scene(node: CaptureNode, scene, root: str, n_frames: int, nav_timeout: float):
    x, y, z = scene.viewpoint
    yaw = math.radians(scene.viewpoint_yaw_deg)
    t0 = time.monotonic()
    deadline = time.monotonic() + 15.0
    while node.get_telemetry() is None:
        if time.monotonic() > deadline:
            raise RuntimeError("No /action/telemetry: is the action layer running?")
        time.sleep(0.2)
    if not node.get_telemetry()["in_air"]:
        print(f"[capture] taking off to {z:.1f} m")
        node.command("TAKEOFF", z)
        wait_takeoff(node.get_telemetry, z, timeout=90.0)
    print(f"[capture] scene {scene.id}: flying to ({x:.2f}, {y:.2f}, {z:.1f}) heading {scene.viewpoint_yaw_deg:.1f} deg")
    node.navigate(x, y, z, yaw, nav_timeout)
    settle_s = node.wait_settled(30.0)
    frames = node.grab(n_frames, 30.0)
    out = scene_dir(root, scene.id)
    records = []
    for k, (_, bgr, depth, pos_ned, q, heading_ned) in enumerate(frames):
        files = save_frame(out, k, bgr, depth)
        cam_t, cam_r = camera_pose_world(pos_ned, q, node.mount_xyz, node.mount_pitch)
        records.append({**files,
                        "position_map": round_list(ned_to_enu(pos_ned), 3),
                        "heading_map_deg": round(math.degrees(yaw_ned_to_enu(heading_ned)), 2),
                        "q_ned_wxyz": round_list(q, 5),
                        "camera_position_map": round_list(cam_t, 3),
                        "camera_rotation_map": [round_list(row, 5) for row in cam_r]})
    first = records[0]["position_map"]
    heading_err = (records[0]["heading_map_deg"] - scene.viewpoint_yaw_deg + 180.0) % 360.0 - 180.0
    meta = {"scene_id": scene.id,
            "planned_viewpoint_map": list(scene.viewpoint),
            "planned_heading_map_deg": scene.viewpoint_yaw_deg,
            "px4_offset_from_plan_m": round(horizontal_offset(scene.viewpoint, first), 3),
            "px4_altitude_error_m": round(first[2] - z, 3),
            "px4_heading_error_deg": round(heading_err, 2),
            "settle_s": round(settle_s, 2),
            "capture_s": round(time.monotonic() - t0, 1),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "camera_mount_xyz_flu": node.mount_xyz,
            "camera_mount_pitch_rad": node.mount_pitch,
            "rgb_camera_info": camera_info_dict(node.rgb_info) if node.rgb_info else None,
            "depth_camera_info": camera_info_dict(node.depth_info) if node.depth_info else None,
            "frames": records}
    write_meta(out, meta)
    print(f"[capture] scene {scene.id}: {len(records)} frames in {out}; PX4 position "
          f"{meta['px4_offset_from_plan_m']} m from the plan, heading error {meta['px4_heading_error_deg']} deg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--scenes", default="/experiments/paper/scenes.json")
    ap.add_argument("--out", default="/experiments/paper/captures")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--nav-timeout", type=float, default=120.0)
    # Must match perception_node camera.mount_* in params.yaml and CAMERA_PITCH_RAD in compose.yaml.
    ap.add_argument("--mount-xyz", type=float, nargs=3, default=[0.132, 0.0, 0.261])
    ap.add_argument("--mount-pitch", type=float, default=0.785)
    ap.add_argument("--rtl", action="store_true", help="return to launch and land after capturing")
    a = ap.parse_args()

    scene = next((s for s in load(a.scenes) if s.id == a.scene), None)
    if scene is None:
        print(f"[capture] no scene {a.scene} in {a.scenes}", file=sys.stderr)
        return 1
    rclpy.init()
    node = CaptureNode(a.mount_xyz, a.mount_pitch)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        capture_scene(node, scene, a.out, a.frames, a.nav_timeout)
        if a.rtl:
            print("[capture] returning to launch")
            node.command("RTL")
            wait_landed(node.get_telemetry, timeout=180.0, what="Return to launch")
        return 0
    except Exception as e:  # noqa: BLE001 - report any failure as one line for the batch script
        print(f"[capture] scene {a.scene} FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
