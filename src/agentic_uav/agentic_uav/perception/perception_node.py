#!/usr/bin/env python3
"""
Perception Layer Node.
Ingests simulated sensor feeds (RGB, Depth, CameraInfo, Drone Odometry), runs YOLOv11,
constructs a probabilistic 3D semantic scene graph, and publishes ROS 2 graph messages and markers.
"""

import math
import time
import re
import threading
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Quaternion

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

from agentic_uav_interfaces.msg import ProbabilisticSceneGraph, SceneNode, SceneEdge, BoundingBox3D
from agentic_uav.perception.yolo_detector import YOLODetector
from agentic_uav.perception.scene_graph_builder import SceneGraphBuilder


class GazeboWorldMonitor:
    """Continuously monitors Gazebo world poses to dynamically discover spawned models."""
    def __init__(self, logger=None):
        self.logger = logger
        self.drone_pos = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        self.models = {}
        self.running = True
        self.last_update = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                models_found = {}
                now = time.time()

                # Method 1: Try reading gz topic pose/info
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
                            models_found[name] = np.array([x, y, z], dtype=np.float32)
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
                                    models_found[name] = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=np.float32)
                            except Exception:
                                pass
                    except Exception:
                        pass

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


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # Declare parameters
        self.declare_parameter("yolo_model", "yolo11n.pt")
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("odom_topic", "/uav/odometry")
        self.declare_parameter("publish_rate_hz", 10.0)

        model_path = self.get_parameter("yolo_model").get_parameter_value().string_value
        conf_thresh = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        cam_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        cam_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.cv_bridge = CvBridge() if CvBridge is not None else None
        self.detector = YOLODetector(model_path=model_path, confidence_threshold=conf_thresh)
        self.graph_builder = SceneGraphBuilder()
        self.gz_monitor = GazeboWorldMonitor(logger=self.get_logger())

        # Cache sensor states
        self.latest_rgb: np.ndarray = None
        self.latest_depth: np.ndarray = None
        self.drone_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.drone_rot_mat = np.eye(3, dtype=np.float32)
        self.last_odom_time = 0.0

        # Subscribers
        self.sub_rgb = self.create_subscription(Image, cam_topic, self.rgb_callback, 10)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.sub_cam_info = self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.sub_odom_model = self.create_subscription(Odometry, "/model/agentic_uav/odometry", self.odom_callback, 10)

        # Publishers
        self.pub_scene_graph = self.create_publisher(ProbabilisticSceneGraph, "/perception/scene_graph", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/perception/scene_graph_markers", 10)

        # Processing loop timer
        rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / rate, self.perception_loop)

        self.get_logger().info(f"Perception Layer initialized. Subscribed to {cam_topic} and {depth_topic}")

    def cam_info_callback(self, msg: CameraInfo):
        fx = msg.k[0] if msg.k[0] > 0 else 554.25
        fy = msg.k[4] if msg.k[4] > 0 else 554.25
        cx = msg.k[2] if msg.k[2] > 0 else 320.0
        cy = msg.k[5] if msg.k[5] > 0 else 240.0
        self.graph_builder.set_camera_intrinsics(fx, fy, cx, cy)

    def rgb_callback(self, msg: Image):
        if self.cv_bridge:
            try:
                self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return
            except Exception:
                pass
        # Fallback raw buffer conversion
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            self.latest_rgb = np.ascontiguousarray(arr.reshape((msg.height, msg.width, -1))[:, :, :3])
        except Exception:
            pass

    def depth_callback(self, msg: Image):
        if self.cv_bridge:
            try:
                self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return
            except Exception:
                pass
        try:
            if "32FC1" in msg.encoding or msg.step == msg.width * 4:
                arr = np.frombuffer(msg.data, dtype=np.float32)
                self.latest_depth = arr.reshape((msg.height, msg.width))
            elif "16UC1" in msg.encoding:
                arr = np.frombuffer(msg.data, dtype=np.uint16)
                self.latest_depth = arr.reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
        except Exception:
            pass

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.drone_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.last_odom_time = time.time()

        # Convert quaternion to rotation matrix
        q = msg.pose.pose.orientation
        self.drone_rot_mat = self._quaternion_to_rot_matrix(q.x, q.y, q.z, q.w)

    def perception_loop(self):
        # Update drone pose from Gazebo ground truth if odom is silent
        if (time.time() - self.last_odom_time) > 0.5 and self.gz_monitor.last_update > 0:
            self.drone_pos = self.gz_monitor.drone_pos.copy()

        # Synchronize all Gazebo world entities (including copy-pasted survivors and fires)
        for name, data in self.gz_monitor.models.items():
            self.graph_builder.update_world_entity(name, data["label"], data["pos"])

        # Run YOLOv11 if camera frames are available
        detections = []
        if self.latest_rgb is not None:
            detections = self.detector.detect(self.latest_rgb)

        if detections:
            nodes_data, edges_data = self.graph_builder.process_detections(
                detections, self.latest_depth, self.drone_pos, self.drone_rot_mat
            )
        else:
            nodes_data, edges_data = self.graph_builder.get_current_graph()

        # Construct ROS message
        graph_msg = ProbabilisticSceneGraph()
        graph_msg.header.stamp = self.get_clock().now().to_msg()
        graph_msg.header.frame_id = "map"
        graph_msg.drone_pose.position.x = float(self.drone_pos[0])
        graph_msg.drone_pose.position.y = float(self.drone_pos[1])
        graph_msg.drone_pose.position.z = float(self.drone_pos[2])
        graph_msg.mission_context = "Search & Rescue Autonomous Survey"

        for n in nodes_data:
            s_node = SceneNode()
            s_node.id = n["id"]
            s_node.label = n["label"]
            s_node.confidence = float(n["confidence"])
            s_node.existence_prob = float(n["existence_prob"])
            s_node.position.x = n["position"][0]
            s_node.position.y = n["position"][1]
            s_node.position.z = n["position"][2]

            s_node.bbox.center.x = n["position"][0]
            s_node.bbox.center.y = n["position"][1]
            s_node.bbox.center.z = n["position"][2]
            s_node.bbox.size.x = n["bbox_size"][0]
            s_node.bbox.size.y = n["bbox_size"][1]
            s_node.bbox.size.z = n["bbox_size"][2]
            s_node.bbox.covariance = n["covariance"]

            s_node.velocity.x = n["velocity"][0]
            s_node.velocity.y = n["velocity"][1]
            s_node.velocity.z = n["velocity"][2]
            s_node.attributes = n["attributes"]
            s_node.observation_count = n["observation_count"]
            s_node.last_observed = self.get_clock().now().to_msg()
            graph_msg.nodes.append(s_node)

        for e in edges_data:
            s_edge = SceneEdge()
            s_edge.source_id = e["source_id"]
            s_edge.target_id = e["target_id"]
            s_edge.relation = e["relation"]
            s_edge.probability = float(e["probability"])
            s_edge.distance = float(e["distance"])
            graph_msg.edges.append(s_edge)

        self.pub_scene_graph.publish(graph_msg)
        self._publish_visualization_markers(graph_msg)

    def _publish_visualization_markers(self, graph: ProbabilisticSceneGraph):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for idx, node in enumerate(graph.nodes):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "semantic_nodes"
            m.id = node.id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position = node.position
            m.scale.x = max(0.4, node.bbox.size.x)
            m.scale.y = max(0.4, node.bbox.size.y)
            m.scale.z = max(0.4, node.bbox.size.z)

            # Color code by semantic type
            if "survivor" in node.label:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.8
            elif "hazard" in node.label:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.1, 0.1, 0.8
            elif "landing" in node.label:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.3, 0.9, 0.8
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.8, 0.2, 0.7

            marker_array.markers.append(m)

            # Text label
            text_m = Marker()
            text_m.header.frame_id = "map"
            text_m.header.stamp = stamp
            text_m.ns = "semantic_labels"
            text_m.id = node.id + 1000
            text_m.type = Marker.TEXT_VIEW_FACING
            text_m.action = Marker.ADD
            text_m.pose.position.x = node.position.x
            text_m.pose.position.y = node.position.y
            text_m.pose.position.z = node.position.z + m.scale.z * 0.7
            text_m.scale.z = 0.35
            text_m.text = f"{node.label} [P={node.existence_prob:.2f}]"
            text_m.color.r, text_m.color.g, text_m.color.b, text_m.color.a = 1.0, 1.0, 1.0, 1.0
            marker_array.markers.append(text_m)

        self.pub_markers.publish(marker_array)

    def _quaternion_to_rot_matrix(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array([
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
        ], dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
