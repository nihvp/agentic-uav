"""Perception layer: camera + depth + PX4 pose -> probabilistic scene graph.

Uses sensor data only. Each new RGB frame is processed once: detect objects, take
their depth from the depth image, place them in the ENU map frame using the PX4 pose
captured when the frame arrived plus the camera mount, and update the scene graph.

Frames arrive with some render/bridge lag, so a pose snapshot is only approximately
the pose at exposure. That is fine while hovering or translating slowly, but during a
turn it rotates every detection around the vehicle, so frames are skipped while the
yaw rate exceeds `max_yaw_rate`.
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

from agentic_uav_interfaces.msg import ProbabilisticSceneGraph, SceneEdge, SceneNode
from agentic_uav.perception.camera_model import Intrinsics, bbox_point, camera_pose_world
from agentic_uav.perception.detector import make_detector
from agentic_uav.perception.motion import YawRateEstimator
from agentic_uav.perception.scene_graph_builder import SceneGraphBuilder

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.detector = make_detector(self.declare_parameter("detector", "color").value)
        rate = self.declare_parameter("rate_hz", 5.0).value
        # Must match the camera mount the sim applies (docker/sim_entrypoint.sh).
        self.mount_xyz = list(self.declare_parameter("camera.mount_xyz", [0.132, 0.0, 0.261]).value)
        self.mount_pitch = float(self.declare_parameter("camera.mount_pitch", 0.785).value)
        topic_lpos = self.declare_parameter("px4.local_position_topic", "/fmu/out/vehicle_local_position").value
        topic_att = self.declare_parameter("px4.attitude_topic", "/fmu/out/vehicle_attitude").value
        self.max_yaw_rate = float(self.declare_parameter("max_yaw_rate", 0.15).value)

        self.bridge = CvBridge()
        self.builder = SceneGraphBuilder()
        self.rgb = None
        self.depth = None
        self.rgb_k = None
        self.depth_k = None
        self.pos_ned = None
        self.q = None
        self.yaw = YawRateEstimator()
        self.frames_used = 0
        self.frames_skipped = 0
        self.frame = None   # (bgr, depth, pos_ned, q, yaw_rate) captured on arrival; None once used

        self.create_subscription(Image, "/camera/image", self._on_rgb, 1)
        self.create_subscription(Image, "/camera/depth", self._on_depth, 1)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_rgb_info, 1)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self._on_depth_info, 1)
        self.create_subscription(VehicleLocalPosition, topic_lpos, self._on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, topic_att, self._on_att, PX4_QOS)
        self.pub = self.create_publisher(ProbabilisticSceneGraph, "/perception/scene_graph", 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"Perception layer up (detector={type(self.detector).__name__}, "
                               f"camera pitch={self.mount_pitch:.2f} rad).")

    def _on_rgb(self, msg: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame = (self.rgb, self.depth, self.pos_ned, self.q, self.yaw.rate())

    def _on_depth(self, msg: Image):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _on_rgb_info(self, msg: CameraInfo):
        self.rgb_k = Intrinsics.from_camera_info(msg)
        self.builder.set_camera_intrinsics(self.rgb_k.fx, self.rgb_k.fy, self.rgb_k.cx, self.rgb_k.cy)

    def _on_depth_info(self, msg: CameraInfo):
        self.depth_k = Intrinsics.from_camera_info(msg)

    def _on_lpos(self, msg: VehicleLocalPosition):
        self.pos_ned = (msg.x, msg.y, msg.z) if (msg.xy_valid and msg.z_valid) else None
        self.yaw.add(msg.timestamp, msg.heading)

    def _on_att(self, msg: VehicleAttitude):
        self.q = tuple(msg.q)

    def _missing_inputs(self):
        names = {"/camera/image": self.rgb, "/camera/depth": self.depth,
                 "/camera/camera_info": self.rgb_k, "/camera/depth/camera_info": self.depth_k,
                 "valid PX4 local position": self.pos_ned, "PX4 attitude": self.q}
        return [n for n, v in names.items() if v is None]

    def _tick(self):
        missing = self._missing_inputs()
        frame, self.frame = self.frame, None
        if missing:
            self.get_logger().warn(f"Waiting for: {', '.join(missing)}.", throttle_duration_sec=10.0)
            nodes, edges = self.builder.get_current_graph()
        elif frame is None or any(v is None for v in frame[1:4]) or abs(frame[4]) > self.max_yaw_rate:
            # No new frame, or taken mid-turn: publish the graph unchanged (no updates, no decay).
            if frame is not None:
                self.frames_skipped += 1
            nodes, edges = self.builder.get_current_graph()
        else:
            bgr, depth, pos_ned, q, _ = frame
            self.frames_used += 1
            cam_pos, cam_rot = camera_pose_world(pos_ned, q, self.mount_xyz, self.mount_pitch)
            detections = self.detector.detect(bgr)
            for d in detections:
                d["point_cam"] = bbox_point(d["bbox"], depth, self.rgb_k, self.depth_k)
            nodes, edges = self.builder.process_detections(detections, cam_pos, cam_rot)
        self.pub.publish(self._to_msg(nodes, edges))
        self.get_logger().info(
            f"Frames used {self.frames_used}, skipped mid-turn {self.frames_skipped}; tracking "
            f"{[n['label'] for n in nodes]}.", throttle_duration_sec=15.0)

    def _to_msg(self, nodes, edges) -> ProbabilisticSceneGraph:
        msg = ProbabilisticSceneGraph()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for n in nodes:
            s = SceneNode()
            s.id = n["id"]
            s.label = n["label"]
            s.confidence = float(n["confidence"])
            s.existence_prob = float(n["existence_prob"])
            s.position.x, s.position.y, s.position.z = (float(v) for v in n["position"])
            s.bbox.center.x, s.bbox.center.y, s.bbox.center.z = (float(v) for v in n["position"])
            s.bbox.size.x, s.bbox.size.y, s.bbox.size.z = (float(v) for v in n["bbox_size"])
            s.bbox.covariance = [float(v) for v in n["covariance"]]
            s.attributes = n["attributes"]
            s.observation_count = n["observation_count"]
            msg.nodes.append(s)
        for e in edges:
            msg.edges.append(SceneEdge(source_id=e["source_id"], target_id=e["target_id"],
                                       relation=e["relation"], probability=float(e["probability"]),
                                       distance=float(e["distance"])))
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
