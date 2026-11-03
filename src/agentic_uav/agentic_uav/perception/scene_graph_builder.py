"""
Probabilistic 3D Semantic Scene Graph Builder.
Transforms 2D detections with measured depth into 3D world-referenced semantic
entities with Bayesian existence estimation, 3D covariance, and relational edges.
Detections without a valid depth are dropped; positions are never guessed.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class TrackedEntity:
    """Internal representation of a 3D semantic entity with Bayesian state."""
    def __init__(self, entity_id: int, label: str, pos_world: np.ndarray, bbox_size: np.ndarray, confidence: float,
                 source: str = ""):
        self.id = entity_id
        self.source = source
        self.label = label
        self.pos_world = pos_world  # [x, y, z] in world frame
        self.bbox_size = bbox_size  # [sx, sy, sz]
        self.confidence = confidence
        self.velocity = np.zeros(3, dtype=np.float32)

        # Log-odds representation of existence probability: L = log(p / (1 - p))
        p_init = np.clip(confidence, 0.05, 0.95)
        self.log_odds = np.log(p_init / (1.0 - p_init))

        # Spatial covariance (3x3)
        self.covariance = np.diag([0.25, 0.25, 0.5])  # default [x, y, z] variance

        self.observation_count = 1
        self.missed_detections = 0
        self.attributes: List[str] = []

    def get_existence_probability(self) -> float:
        """Converts log-odds back to probability [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def update_bayesian_existence(self, hit: bool, p_detection_given_exists: float = 0.85, p_false_alarm: float = 0.1):
        """Recursive Bayesian log-odds update for object persistence."""
        if hit:
            delta = np.log(p_detection_given_exists / p_false_alarm)
            self.missed_detections = 0
            self.observation_count += 1
        else:
            delta = np.log((1.0 - p_detection_given_exists) / (1.0 - p_false_alarm))
            self.missed_detections += 1

        self.log_odds = np.clip(self.log_odds + delta, -5.0, 5.0)

    def update_position(self, new_pos: np.ndarray, alpha: float = 0.3):
        """Exponential moving average filter for 3D centroid."""
        self.velocity = (new_pos - self.pos_world) * alpha
        self.pos_world = (1.0 - alpha) * self.pos_world + alpha * new_pos


class SceneGraphBuilder:
    """Builds and updates a probabilistic 3D semantic scene graph."""

    def __init__(self, camera_fx: float = 554.25, camera_fy: float = 554.25,
                 camera_cx: float = 320.0, camera_cy: float = 240.0,
                 max_association_distance: float = 2.5):
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = camera_cx
        self.cy = camera_cy
        self.max_association_distance = max_association_distance

        self.next_entity_id = 1
        self.tracked_entities: Dict[int, TrackedEntity] = {}

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def get_current_graph(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Returns the current nodes and edges directly without requiring a new camera frame."""
        nodes = []
        for t_id, track in self.tracked_entities.items():
            nodes.append({
                "id": track.id,
                "label": track.label,
                "confidence": float(track.confidence),
                "existence_prob": float(track.get_existence_probability()),
                "position": track.pos_world.tolist(),
                "bbox_size": track.bbox_size.tolist(),
                "velocity": track.velocity.tolist(),
                "covariance": track.covariance.flatten().tolist(),
                "observation_count": track.observation_count,
                "attributes": self._derive_attributes(track)
            })
        edges = self._generate_relational_edges(nodes)
        return nodes, edges

    def deproject_pixel_to_3d_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """Pinhole camera inverse projection: (u, v, depth along optical axis) -> (X_c, Y_c, Z_c)."""
        x_c = (u - self.cx) * depth / self.fx
        y_c = (v - self.cy) * depth / self.fy
        z_c = depth
        return np.array([x_c, y_c, z_c], dtype=np.float32)

    def transform_camera_to_world(self, pt_cam: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray) -> np.ndarray:
        """Optical frame (X-right, Y-down, Z-forward) -> ENU world, given the camera pose
        (see perception/camera_model.camera_pose_world)."""
        return cam_rot @ pt_cam + cam_pos

    def process_detections(self, detections: List[Dict[str, Any]],
                           cam_pos: np.ndarray,
                           cam_rot: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Ingests 2D detections that carry a measured "point_cam" (3D point in the camera
        optical frame, from the depth image), places them in the world, performs data
        association, updates track state, and generates graph nodes and edges.
        """
        current_frame_entities = []

        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            label = det["label"]
            conf = det["confidence"]
            pt_cam = det.get("point_cam")
            if pt_cam is None or not np.all(np.isfinite(pt_cam)) or pt_cam[2] <= 0.1:
                continue
            pt_cam = np.asarray(pt_cam, dtype=float)
            depth_val = float(pt_cam[2])

            pt_world = self.transform_camera_to_world(pt_cam, cam_pos, cam_rot)

            # Estimate 3D bounding box size from 2D pixel extent and depth
            box_w_cam = abs(bbox[2] - bbox[0]) * depth_val / self.fx
            box_h_cam = abs(bbox[3] - bbox[1]) * depth_val / self.fy
            box_d_cam = max(0.5, box_w_cam * 0.8)
            bbox_size = np.array([max(0.4, box_w_cam), max(0.4, box_d_cam), max(0.4, box_h_cam)])

            # Depth-dependent covariance: sigma_z grows quadratically with distance
            sigma_xy = 0.1 * depth_val
            sigma_z = 0.05 * (depth_val ** 1.8)
            cov_matrix = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])

            current_frame_entities.append({
                "label": label,
                "confidence": conf,
                "pos_world": pt_world,
                "source": det.get("source", ""),
                "bbox_size": bbox_size,
                "covariance": cov_matrix
            })

        # Data Association (Nearest-Neighbor Euclidean)
        matched_track_ids = set()
        for obs in current_frame_entities:
            best_id = None
            min_dist = float("inf")

            for t_id, track in self.tracked_entities.items():
                if track.label == obs["label"]:
                    dist = np.linalg.norm(track.pos_world - obs["pos_world"])
                    if dist < min_dist and dist < self.max_association_distance:
                        min_dist = dist
                        best_id = t_id

            if best_id is not None:
                # Update existing track
                track = self.tracked_entities[best_id]
                track.update_position(obs["pos_world"])
                track.confidence = (track.confidence + obs["confidence"]) / 2.0
                track.covariance = obs["covariance"]
                track.update_bayesian_existence(hit=True)
                matched_track_ids.add(best_id)
            else:
                # Spawn new entity
                new_id = self.next_entity_id
                self.next_entity_id += 1
                new_track = TrackedEntity(
                    entity_id=new_id,
                    label=obs["label"],
                    pos_world=obs["pos_world"],
                    bbox_size=obs["bbox_size"],
                    confidence=obs["confidence"],
                    source=obs["source"]
                )
                new_track.covariance = obs["covariance"]
                self.tracked_entities[new_id] = new_track
                matched_track_ids.add(new_id)

        # Decay non-matched tracks
        to_delete = []
        for t_id, track in self.tracked_entities.items():
            if t_id not in matched_track_ids:
                track.update_bayesian_existence(hit=False)
                if track.get_existence_probability() < 0.15 or track.missed_detections > 15:
                    to_delete.append(t_id)

        for t_id in to_delete:
            del self.tracked_entities[t_id]

        # Convert tracks to Graph Nodes
        nodes = []
        for t_id, track in self.tracked_entities.items():
            nodes.append({
                "id": track.id,
                "label": track.label,
                "confidence": float(track.confidence),
                "existence_prob": float(track.get_existence_probability()),
                "position": track.pos_world.tolist(),
                "bbox_size": track.bbox_size.tolist(),
                "velocity": track.velocity.tolist(),
                "covariance": track.covariance.flatten().tolist(),
                "observation_count": track.observation_count,
                "attributes": self._derive_attributes(track)
            })

        # Derive Topological & Relational Graph Edges
        edges = self._generate_relational_edges(nodes)
        return nodes, edges

    def _derive_attributes(self, track: TrackedEntity) -> List[str]:
        attrs = ["state:confirmed" if track.get_existence_probability() > 0.7 else "state:tentative"]
        if track.source:
            attrs.append(f"detector:{track.source}")
        if track.label == "survivor":
            attrs.extend(["priority:high", "action:assist"])
        elif track.label == "fire_hazard":
            attrs.extend(["hazard_level:critical", "action:evacuate_zone"])
        elif track.label == "landing_zone":
            attrs.append("status:open")
        return attrs

    def _generate_relational_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Computes spatial, containment, and hazard relational edges between entities."""
        edges = []
        num_nodes = len(nodes)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                n1 = nodes[i]
                n2 = nodes[j]

                p1 = np.array(n1["position"])
                p2 = np.array(n2["position"])
                dist = float(np.linalg.norm(p1 - p2))

                # Proximity Edge (Spatial Neigborhood)
                if dist < 6.0:
                    prob_near = float(np.exp(- (dist ** 2) / (2.0 * (3.0 ** 2))))
                    edges.append({
                        "source_id": n1["id"],
                        "target_id": n2["id"],
                        "relation": "near",
                        "probability": prob_near,
                        "distance": dist
                    })

                # Hazard Threat Edge
                if (n1["label"] == "fire_hazard" and n2["label"] == "survivor") or \
                   (n2["label"] == "fire_hazard" and n1["label"] == "survivor"):
                    source_id = n1["id"] if n1["label"] == "fire_hazard" else n2["id"]
                    target_id = n2["id"] if n1["label"] == "fire_hazard" else n1["id"]
                    threat_prob = float(np.clip(1.0 - (dist / 12.0), 0.1, 1.0))
                    edges.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation": "threatens",
                        "probability": threat_prob,
                        "distance": dist
                    })

                # Obstacle Blocking Landing Zone Edge
                if ("debris" in n1["label"] and n2["label"] == "landing_zone") or \
                   ("debris" in n2["label"] and n1["label"] == "landing_zone"):
                    edges.append({
                        "source_id": n1["id"],
                        "target_id": n2["id"],
                        "relation": "blocking_path",
                        "probability": 0.85 if dist < 3.0 else 0.4,
                        "distance": dist
                    })

        return edges
