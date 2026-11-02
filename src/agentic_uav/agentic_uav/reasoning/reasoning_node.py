"""Reasoning layer: runs LLM-driven missions over real ROS tools.

Start a mission:
  ros2 service call /reasoning/run_mission agentic_uav_interfaces/srv/RunMission \
    "{goal: 'Take off to 5 m, fly to (8, 4, 5), then return to launch.'}"
"""

import json
import math
import threading
import time
from typing import Any, Dict

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from agentic_uav_interfaces.action import NavigateTo
from agentic_uav_interfaces.msg import FlightTelemetry, ProbabilisticSceneGraph
from agentic_uav_interfaces.srv import FlightCommand, RAGQuery, RunMission
from agentic_uav.reasoning.agent_loop import AgentLoop
from agentic_uav.reasoning.llm import make_provider
from agentic_uav.reasoning.prompts import SYSTEM_PROMPT
from agentic_uav.reasoning.tools import ToolRegistry, obj

NUM = {"type": "number"}


def wait_future(future, timeout: float):
    """Block a worker thread until an rclpy future completes (executor runs elsewhere)."""
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    if not done.wait(timeout):
        raise TimeoutError(f"No response within {timeout:.0f} s.")
    return future.result()


class ReasoningNode(Node):
    def __init__(self):
        super().__init__("reasoning_node")
        provider_name = self.declare_parameter("llm.provider", "gemini").value
        model = self.declare_parameter("llm.model", "gemini-2.5-flash").value
        self.max_steps = self.declare_parameter("max_steps", 20).value
        self.service_timeout = self.declare_parameter("service_timeout_s", 10.0).value
        self.nav_timeout = self.declare_parameter("nav_timeout_s", 180.0).value

        self.scene: Dict[str, Any] = {"nodes": [], "edges": [], "stamp": None}
        self.telemetry: Dict[str, Any] = {}
        self._busy = threading.Lock()
        cb = ReentrantCallbackGroup()

        self.create_subscription(ProbabilisticSceneGraph, "/perception/scene_graph", self._on_scene, 10)
        self.create_subscription(FlightTelemetry, "/action/telemetry", self._on_telemetry, 10)
        self.nav_client = ActionClient(self, NavigateTo, "/action/navigate_to", callback_group=cb)
        self.flight_client = self.create_client(FlightCommand, "/action/flight_command", callback_group=cb)
        self.rag_client = self.create_client(RAGQuery, "/learning/rag_query", callback_group=cb)
        self.create_service(RunMission, "/reasoning/run_mission", self._on_run_mission, callback_group=cb)

        registry = ToolRegistry()
        self._register_tools(registry)
        self.agent = AgentLoop(
            make_provider(provider_name, model=model), registry, SYSTEM_PROMPT,
            max_steps=self.max_steps,
            on_event=lambda e: self.get_logger().info(
                f"[step {e['step']}] {e['tool']}({e['arguments']}) -> ok={e['ok']}: {e['result']}"),
        )
        self.get_logger().info(f"Reasoning layer up (provider={provider_name}, model={model}).")

    # ---- state ----------------------------------------------------------

    def _on_scene(self, msg: ProbabilisticSceneGraph):
        self.scene = {
            "stamp": time.time(),
            "nodes": [{"id": n.id, "label": n.label, "existence_prob": round(n.existence_prob, 2),
                       "position": [round(n.position.x, 2), round(n.position.y, 2), round(n.position.z, 2)],
                       "attributes": list(n.attributes)} for n in msg.nodes],
            "edges": [{"source": e.source_id, "target": e.target_id, "relation": e.relation,
                       "probability": round(e.probability, 2), "distance": round(e.distance, 2)}
                      for e in msg.edges],
        }

    def _on_telemetry(self, msg: FlightTelemetry):
        p = msg.current_pose.position
        self.telemetry = {
            "armed": msg.armed, "px4_nav_state": msg.px4_nav_state,
            "controller_state": msg.controller_state, "in_air": msg.is_in_air,
            "position": [round(p.x, 2), round(p.y, 2), round(p.z, 2)],
            "battery_percent": None if math.isnan(msg.battery_percentage) else round(msg.battery_percentage, 1),
            "geofence_ok": msg.geofence_ok,
        }

    # ---- tools ----------------------------------------------------------

    def _register_tools(self, r: ToolRegistry):
        r.register("get_telemetry", "Current vehicle state, position (map frame) and battery.",
                   obj({}), self._tool_telemetry)
        r.register("get_scene_graph", "Objects currently tracked by perception, with relations.",
                   obj({}), self._tool_scene)
        r.register("takeoff", "Arm and take off to an altitude in metres.",
                   obj({"altitude": NUM}, ["altitude"]), lambda altitude: self._flight("TAKEOFF", altitude))
        r.register("navigate_to", "Fly to (x, y, z) in the map frame; blocks until reached or failed.",
                   obj({"x": NUM, "y": NUM, "z": NUM}, ["x", "y", "z"]), self._tool_navigate)
        r.register("hold", "Stop and hold the current position.", obj({}), lambda: self._flight("HOLD"))
        r.register("land", "Land at the current position.", obj({}), lambda: self._flight("LAND"))
        r.register("return_to_launch", "Fly home and land.", obj({}), lambda: self._flight("RTL"))
        r.register("query_memory", "Search procedures and past mission reports.",
                   obj({"query": {"type": "string"}}, ["query"]), self._tool_memory)

    def _tool_telemetry(self):
        if not self.telemetry:
            raise RuntimeError("No telemetry received from the action layer.")
        return self.telemetry

    def _tool_scene(self):
        if self.scene["stamp"] is None:
            raise RuntimeError("No scene graph received from the perception layer.")
        return {**self.scene, "age_s": round(time.time() - self.scene["stamp"], 1)}

    def _flight(self, command: str, altitude: float = 0.0):
        if not self.flight_client.wait_for_service(timeout_sec=self.service_timeout):
            raise RuntimeError("Action layer flight_command service unavailable.")
        res = wait_future(self.flight_client.call_async(
            FlightCommand.Request(command=command, altitude=float(altitude))), self.service_timeout)
        if not res.success:
            raise RuntimeError(res.message)
        return res.message

    def _tool_navigate(self, x: float, y: float, z: float):
        if not self.nav_client.wait_for_server(timeout_sec=self.service_timeout):
            raise RuntimeError("Action layer navigate_to server unavailable.")
        goal = NavigateTo.Goal()
        goal.target.x, goal.target.y, goal.target.z = float(x), float(y), float(z)
        goal.yaw = math.nan
        handle = wait_future(self.nav_client.send_goal_async(goal), self.service_timeout)
        if not handle.accepted:
            raise RuntimeError("Navigation goal rejected.")
        res = wait_future(handle.get_result_async(), self.nav_timeout).result
        if not res.success:
            raise RuntimeError(res.message)
        p = res.final_position
        return {"message": res.message, "final_position": [round(p.x, 2), round(p.y, 2), round(p.z, 2)]}

    def _tool_memory(self, query: str):
        if not self.rag_client.wait_for_service(timeout_sec=self.service_timeout):
            raise RuntimeError("Learning layer rag_query service unavailable.")
        res = wait_future(self.rag_client.call_async(
            RAGQuery.Request(query_text=query, top_k=3, filter_category="")), self.service_timeout)
        if not res.success:
            raise RuntimeError(res.error_message)
        return [{"id": i, "score": round(s, 3), "content": c}
                for i, c, s in zip(res.document_ids, res.contents, res.similarity_scores)]

    # ---- mission service ------------------------------------------------

    def _on_run_mission(self, request, response):
        if not self._busy.acquire(blocking=False):
            response.success, response.summary = False, "A mission is already running."
            return response
        try:
            self.get_logger().info(f"Mission: {request.goal}")
            result = self.agent.run(request.goal, max_steps=request.max_steps if request.max_steps > 0 else None)
            response.success, response.summary = result.success, result.summary
            response.trace_json = json.dumps(result.trace, default=str)
            self.get_logger().info(f"Mission finished: success={result.success} - {result.summary}")
        except Exception as ex:
            response.success, response.summary = False, f"Mission error: {ex}"
        finally:
            self._busy.release()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ReasoningNode()
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
