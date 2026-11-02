#!/usr/bin/env python3
"""
Reasoning Layer Node.
Cognitive engine node powered by Gemini API executing a ReAct (Reasoning and Acting) workflow.
Subscribes to the 3D Probabilistic Semantic Scene Graph and vehicle telemetry,
interacts with the RAG Learning Layer and MCP Integration Layer, and publishes symbolic action commands.
"""

import json
import uuid
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Point

from agentic_uav_interfaces.msg import ProbabilisticSceneGraph, FlightTelemetry, AgentCommand
from agentic_uav_interfaces.srv import TriggerReActCycle, RAGQuery, ExecuteMCPTool
from agentic_uav.reasoning.gemini_client import GeminiClient
from agentic_uav.reasoning.react_engine import ReActEngine


class ReasoningNode(Node):
    def __init__(self):
        super().__init__("reasoning_node")

        # Declare parameters
        self.declare_parameter("gemini_api_key", "")
        self.declare_parameter("gemini_model", "gemini-2.5-flash")
        self.declare_parameter("default_mission_goal", "Search and rescue reconnaissance over Sector Alpha: find survivors, identify hazards, verify clear landing zones.")
        self.declare_parameter("cognitive_cycle_rate_hz", 0.5)  # Run ReAct every 2.0s
        self.declare_parameter("auto_cycle_enabled", False)

        api_key = self.get_parameter("gemini_api_key").get_parameter_value().string_value
        model_name = self.get_parameter("gemini_model").get_parameter_value().string_value
        self.mission_goal = self.get_parameter("default_mission_goal").get_parameter_value().string_value
        self.auto_cycle = self.get_parameter("auto_cycle_enabled").get_parameter_value().bool_value

        self.gemini_client = GeminiClient(api_key=api_key, model_name=model_name)
        self.react_engine = ReActEngine(self.gemini_client)

        # State storage
        self.latest_scene_nodes = []
        self.latest_scene_edges = []
        self.latest_telemetry = {
            "armed": True,
            "mode": "OFFBOARD",
            "battery": 92.5,
            "altitude": 5.0,
            "geofence_ok": True
        }
        self.last_observation = None
        self.cycle_count = 0

        # Subscriptions
        self.sub_graph = self.create_subscription(
            ProbabilisticSceneGraph, "/perception/scene_graph", self.scene_graph_callback, 10
        )
        self.sub_telem = self.create_subscription(
            FlightTelemetry, "/action/flight_telemetry", self.telemetry_callback, 10
        )

        # Publisher for symbolic flight and digital tool commands
        self.pub_command = self.create_publisher(AgentCommand, "/action/agent_command", 10)

        # Service Servers
        self.srv_trigger = self.create_service(
            TriggerReActCycle, "/reasoning/trigger_react_cycle", self.handle_trigger_cycle
        )

        # Service Clients for Layer Integration
        self.cli_rag = self.create_client(RAGQuery, "/learning/rag_query")
        self.cli_mcp = self.create_client(ExecuteMCPTool, "/integration/execute_mcp_tool")

        # Autonomous reasoning loop
        rate = self.get_parameter("cognitive_cycle_rate_hz").get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / max(0.1, rate), self.cognitive_cycle_tick)

        self.get_logger().info(f"Reasoning Layer active with goal: '{self.mission_goal}'")

    def scene_graph_callback(self, msg: ProbabilisticSceneGraph):
        nodes = []
        for n in msg.nodes:
            nodes.append({
                "id": n.id,
                "label": n.label,
                "confidence": n.confidence,
                "existence_prob": n.existence_prob,
                "position": [n.position.x, n.position.y, n.position.z],
                "attributes": list(n.attributes)
            })
        edges = []
        for e in msg.edges:
            edges.append({
                "source_id": e.source_id,
                "target_id": e.target_id,
                "relation": e.relation,
                "probability": e.probability,
                "distance": e.distance
            })
        self.latest_scene_nodes = nodes
        self.latest_scene_edges = edges

    def telemetry_callback(self, msg: FlightTelemetry):
        self.latest_telemetry = {
            "armed": msg.armed,
            "mode": msg.flight_mode,
            "battery": msg.battery_percentage,
            "altitude": msg.current_pose.position.z,
            "geofence_ok": msg.geofence_ok
        }

    def handle_trigger_cycle(self, request, response):
        if request.mission_goal:
            self.mission_goal = request.mission_goal
        thought, action, action_input = self._execute_react_cycle()
        response.success = True
        response.final_plan = f"Action: {action} with inputs: {json.dumps(action_input)}"
        response.thought_trace = thought
        return response

    def cognitive_cycle_tick(self):
        if not self.auto_cycle:
            return
        # Run cognitive cycle only if we have at least one scene graph update or initial telemetry
        self.cycle_count += 1
        if self.cycle_count % 3 == 0 or len(self.latest_scene_nodes) > 0:
            self._execute_react_cycle()

    def _execute_react_cycle(self):
        prompt = self.react_engine.format_state_prompt(
            mission_goal=self.mission_goal,
            scene_nodes=self.latest_scene_nodes,
            scene_edges=self.latest_scene_edges,
            telemetry=self.latest_telemetry,
            last_observation=self.last_observation
        )

        thought, action, action_input = self.react_engine.step(prompt)
        self.get_logger().info(f"[ReAct Step]\nThought: {thought}\nAction: {action}\nInput: {action_input}")

        # Dispatch action based on target layer
        if action == "query_rag_memory":
            query_str = action_input.get("query", "standard emergency response procedure")
            self._call_rag_service(query_str)
        elif action == "invoke_mcp_tool":
            tool_name = action_input.get("tool_name", "weather_advisory")
            tool_args = action_input.get("arguments", {})
            self._call_mcp_service(tool_name, tool_args)
            # Concurrently fly the UAV to maintain aerial visual coverage of the incident
            loc = tool_args.get("location", [8.0, 4.0, 4.0])
            flight_input = {
                "x": float(loc[0]) if len(loc) > 0 else 8.0,
                "y": float(loc[1]) if len(loc) > 1 else 4.0,
                "z": float(loc[2]) if len(loc) > 2 and loc[2] > 0.5 else 4.0,
                "speed": 2.0
            }
            self._publish_flight_command("navigate_waypoint", flight_input, thought)
            self.last_observation = f"Invoked MCP tool '{tool_name}' and dispatched aerial reconnaissance to ({flight_input['x']}, {flight_input['y']}, {flight_input['z']})."
        elif action in ["navigate_waypoint", "inspect_target", "deploy_payload", "return_to_launch", "mission_complete"]:
            self._publish_flight_command(action, action_input, thought)
            self.last_observation = f"Dispatched {action} command to Action Layer executor."
        else:
            self.last_observation = f"Unrecognized action '{action}'. Defaulting to hold position."

        return thought, action, action_input

    def _call_rag_service(self, query: str):
        if not self.cli_rag.service_is_ready():
            self.last_observation = f"RAG Memory service not ready. Proceeding with standard onboard heuristic for: {query}"
            return

        req = RAGQuery.Request()
        req.query_text = query
        req.top_k = 2
        req.filter_category = ""

        future = self.cli_rag.call_async(req)
        future.add_done_callback(self._on_rag_response)

    def _on_rag_response(self, future):
        try:
            res = future.result()
            if res.success and len(res.contents) > 0:
                retrieved = " | ".join(res.contents[:2])
                self.last_observation = f"RAG Memory retrieved protocol: {retrieved}"
            else:
                self.last_observation = "RAG Memory query yielded no matching historical documents."
        except Exception as e:
            self.last_observation = f"RAG Memory query exception: {e}"

    def _call_mcp_service(self, tool_name: str, args: dict):
        if not self.cli_mcp.service_is_ready():
            self.last_observation = f"MCP Integration service not ready. Tool '{tool_name}' deferred."
            return

        req = ExecuteMCPTool.Request()
        req.tool_name = tool_name
        req.arguments_json = json.dumps(args)
        req.client_token = "uav_reasoning_node_auth_token"

        future = self.cli_mcp.call_async(req)
        future.add_done_callback(lambda f: self._on_mcp_response(f, tool_name))

    def _on_mcp_response(self, future, tool_name: str):
        try:
            res = future.result()
            if res.success:
                self.last_observation = f"MCP Tool '{tool_name}' executed successfully. Result: {res.result_json}"
            else:
                self.last_observation = f"MCP Tool '{tool_name}' failed: {res.error_message}"
        except Exception as e:
            self.last_observation = f"MCP Tool exception: {e}"

    def _publish_flight_command(self, action: str, action_input: dict, thought: str):
        cmd = AgentCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "map"
        cmd.command_id = str(uuid.uuid4())[:8]
        cmd.action_type = action.upper()
        cmd.reasoning_trace = thought

        if action == "navigate_waypoint":
            cmd.target_position.x = float(action_input.get("x", 0.0))
            cmd.target_position.y = float(action_input.get("y", 0.0))
            cmd.target_position.z = float(action_input.get("z", 3.0))
            cmd.speed = float(action_input.get("speed", 2.0))
        elif action == "inspect_target":
            cmd.target_position.x = float(action_input.get("x", 0.0))
            cmd.target_position.y = float(action_input.get("y", 0.0))
            cmd.target_position.z = float(action_input.get("z", 4.0))
            cmd.loiter_duration = float(action_input.get("duration", 10.0))
            cmd.tool_name = "camera_gimbal_track"
            cmd.tool_parameters = json.dumps({"target_id": action_input.get("target_id", 0)})
        elif action == "deploy_payload":
            cmd.tool_name = "payload_release_actuator"
            cmd.tool_parameters = json.dumps(action_input)
        elif action == "return_to_launch":
            cmd.action_type = "RTL"
            cmd.tool_parameters = json.dumps({"reason": action_input.get("reason", "manual")})

        self.pub_command.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ReasoningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
