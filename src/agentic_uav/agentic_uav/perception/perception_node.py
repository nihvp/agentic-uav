"""Perception layer: camera + depth -> probabilistic scene graph.

Uses sensor data only. Phase 2 work (see docs/REBUILD_PLAN.md):
  * YOLO detector
  * camera pose from PX4 attitude + camera extrinsics (currently identity, so
    positions are NOT valid until that lands; the NullDetector produces none)
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from agentic_uav_interfaces.msg import ProbabilisticSceneGraph, SceneEdge, SceneNode
from agentic_uav.perception.detector import make_detector
from agentic_uav.perception.scene_graph_builder import SceneGraphBuilder


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.detector = make_detector(self.declare_parameter("detector", "none").value)
        rate = self.declare_parameter("rate_hz", 5.0).value
        self.bridge = CvBridge()
        self.builder = SceneGraphBuilder()
        self.rgb = None
        self.depth = None
        self.cam_pos = np.zeros(3, dtype=np.float32)
        self.cam_rot = np.eye(3, dtype=np.float32)

        self.create_subscription(Image, "/camera/image", self._on_rgb, 1)
        self.create_subscription(Image, "/camera/depth", self._on_depth, 1)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, 1)
        self.pub = self.create_publisher(ProbabilisticSceneGraph, "/perception/scene_graph", 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"Perception layer up (detector={type(self.detector).__name__}).")

    def _on_rgb(self, msg: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _on_depth(self, msg: Image):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _on_info(self, msg: CameraInfo):
        self.builder.set_camera_intrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _tick(self):
        if self.rgb is None:
            self.get_logger().warn("No camera frames yet on /camera/image.", throttle_duration_sec=10.0)
            nodes, edges = self.builder.get_current_graph()
        else:
            detections = self.detector.detect(self.rgb)
            nodes, edges = self.builder.process_detections(detections, self.depth, self.cam_pos, self.cam_rot)

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
        self.pub.publish(msg)


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
