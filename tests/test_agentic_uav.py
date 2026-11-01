"""
Unit and Integration Test Suite for the 5-Layer Agentic UAV System.
Tests all five layers in standalone isolation:
1. Perception Layer (YOLOv11 & Probabilistic Scene Graph)
2. Reasoning Layer (Gemini ReAct Engine)
3. Action Layer (PX4 Controller & Actuators)
4. Integration Layer (MCP Registry & Tool Security)
5. Learning Layer (Semantic Vector Store & RAG)
"""

import sys
import os
import json
import numpy as np
import pytest

# Add package source to path for direct testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/agentic_uav")))

from agentic_uav.perception.yolo_detector import YOLODetector
from agentic_uav.perception.scene_graph_builder import SceneGraphBuilder, TrackedEntity
from agentic_uav.reasoning.gemini_client import GeminiClient
from agentic_uav.reasoning.react_engine import ReActEngine
from agentic_uav.action.px4_controller import PX4Controller, FlightState
from agentic_uav.action.tool_actuators import ToolActuators
from agentic_uav.integration.mcp_server_registry import MCPToolRegistry
from agentic_uav.learning.vector_store import SemanticVectorStore
from agentic_uav.learning.default_sop_knowledge import DEFAULT_KNOWLEDGE_DOCUMENTS


# ==============================================================================
# Layer 1: Perception Tests
# ==============================================================================
def test_perception_yolo_synthetic_fallback():
    detector = YOLODetector(model_path="nonexistent_yolo11n.pt")
    # Feed test RGB image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(img)
    assert len(detections) > 0
    assert "bbox" in detections[0]
    assert "label" in detections[0]
    assert "confidence" in detections[0]


def test_scene_graph_builder_projection_and_bayesian():
    builder = SceneGraphBuilder(camera_fx=500.0, camera_fy=500.0, camera_cx=320.0, camera_cy=240.0)
    drone_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    drone_rot = np.eye(3, dtype=np.float32)

    detections = [
        {"bbox": [300, 220, 340, 260], "confidence": 0.85, "label": "survivor"},
        {"bbox": [400, 220, 440, 260], "confidence": 0.90, "label": "fire_hazard"}
    ]
    depth = np.ones((480, 640), dtype=np.float32) * 6.0

    nodes, edges = builder.process_detections(detections, depth, drone_pos, drone_rot)
    assert len(nodes) == 2
    assert nodes[0]["existence_prob"] > 0.5
    assert len(nodes[0]["position"]) == 3

    # Check relational edges
    assert len(edges) > 0
    threat_edge = [e for e in edges if e["relation"] == "threatens"]
    assert len(threat_edge) == 1


# ==============================================================================
# Layer 2: Reasoning Tests
# ==============================================================================
def test_reasoning_react_engine():
    client = GeminiClient(api_key="")  # Triggers offline ReAct simulation
    engine = ReActEngine(client)

    scene_nodes = [{
        "id": 1,
        "label": "survivor",
        "position": [10.0, 4.0, 0.0],
        "existence_prob": 0.92,
        "attributes": ["priority:high"]
    }]
    scene_edges = []
    telemetry = {"armed": True, "mode": "OFFBOARD", "battery": 95.0, "altitude": 5.0}

    prompt = engine.format_state_prompt(
        mission_goal="Search for survivors and secure sector",
        scene_nodes=scene_nodes,
        scene_edges=scene_edges,
        telemetry=telemetry
    )

    thought, action, action_input = engine.step(prompt)
    assert len(thought) > 0
    assert action in ["inspect_target", "navigate_waypoint", "invoke_mcp_tool", "query_rag_memory"]
    assert isinstance(action_input, dict)


# ==============================================================================
# Layer 3: Action Tests
# ==============================================================================
def test_action_px4_controller_and_geofence():
    ctrl = PX4Controller(takeoff_altitude=4.0, max_speed=5.0, geofence_radius=50.0, geofence_max_alt=20.0)
    assert ctrl.arm_and_takeoff() is True
    assert ctrl.state == FlightState.TAKEOFF

    # Step simulation
    target_pos, vel, yaw = ctrl.compute_step(dt=0.1)
    assert target_pos[2] == 4.0

    # Test geofence violation triggering RTL
    ctrl.update_odometry(np.array([100.0, 0.0, 5.0], dtype=np.float32), np.zeros(3))
    assert ctrl.check_geofence() is False
    ctrl.compute_step(dt=0.1)
    assert ctrl.state == FlightState.RTL


def test_action_tool_actuators():
    actuators = ToolActuators()
    ok, msg = actuators.execute_tool("camera_gimbal_track", {"pitch": -60.0})
    assert ok is True
    assert actuators.gimbal_pitch == -60.0

    ok, msg = actuators.execute_tool("payload_release_actuator", {"payload_type": "First-Aid"})
    assert ok is True
    assert actuators.payload_released is True

    # Secondary release should fail (empty magazine)
    ok2, msg2 = actuators.execute_tool("payload_release_actuator", {})
    assert ok2 is False


# ==============================================================================
# Layer 4: Integration Tests (MCP)
# ==============================================================================
def test_mcp_tool_registry():
    registry = MCPToolRegistry(allowed_tokens=["test_token_123"])
    tools = registry.list_tools()
    assert len(tools) >= 4
    tool_names = [t["name"] for t in tools]
    assert "emergency_alert_dispatch" in tool_names
    assert "utm_airspace_clearance" in tool_names

    # Test Unauthorized Token
    ok, _, err = registry.execute_tool("utm_airspace_clearance", {"altitude_ceiling": 20.0}, "bad_token")
    assert ok is False
    assert "Unauthorized" in err

    # Test Authorized Execution
    ok, res_json, err = registry.execute_tool(
        "utm_airspace_clearance",
        {"altitude_ceiling": 20.0},
        "test_token_123"
    )
    assert ok is True
    res = json.loads(res_json)
    assert res["authorization"] == "GRANTED"


# ==============================================================================
# Layer 5: Learning Tests (RAG & Vector Memory)
# ==============================================================================
def test_learning_rag_vector_store():
    store = SemanticVectorStore()
    for doc in DEFAULT_KNOWLEDGE_DOCUMENTS:
        store.insert(
            doc_id=doc["id"],
            content=f"{doc['title']}: {doc['content']}",
            category=doc["category"]
        )

    # Query for landing zone procedures
    results = store.query("What are the clearance rules for an obstructed landing zone?", top_k=2)
    assert len(results) > 0
    top_doc, sim = results[0]
    assert "SOP-LZ-003" in top_doc["id"] or "landing" in top_doc["content"].lower()

    # Query with category filtering
    filtered_results = store.query("battery failsafe", top_k=1, filter_category="REGULATION")
    assert len(filtered_results) > 0
    assert filtered_results[0][0]["category"] == "REGULATION"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
