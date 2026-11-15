"""Reasoning layer: runs LLM-driven missions over real ROS tools.

Start a mission:
  ros2 service call /reasoning/run_mission agentic_uav_interfaces/srv/RunMission \
    "{goal: 'Take off to 5 m, fly to (8, 4, 5), then return to launch.'}"
"""

import json
import math
import os
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
from agentic_uav.reasoning.flight_waits import wait_landed, wait_takeoff
from agentic_uav.reasoning.llm import make_provider
from agentic_uav.reasoning.prompts import SYSTEM_PROMPT
from agentic_uav.reasoning.scan import compact_nodes, merge_sightings, scan_headings
from agentic_uav.reasoning.tools import ToolRegistry, obj

NUM = {"type": "number"}
# Below these battery levels the tools refuse to start new flight segments.
MIN_BATTERY_TAKEOFF = 30.0
MIN_BATTERY_NAVIGATE = 20.0


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
        model = self.declare_parameter("llm.model", "gemini-3.5-flash").value
        rpm = float(self.declare_parameter("llm.requests_per_minute", 0.0).value)
        thinking = self.declare_parameter("llm.thinking_level", "low").value
        llm_timeout = float(self.declare_parameter("llm.timeout_s", 90.0).value)
        self.max_steps = self.declare_parameter("max_steps", 20).value
        self.service_timeout = self.declare_parameter("service_timeout_s", 10.0).value
        self.nav_timeout = self.declare_parameter("nav_timeout_s", 180.0).value
        self.trace_dir = self.declare_parameter("trace_dir", "/data/missions").value
        # After arriving, let perception process a few frames before reading the scene.
        self.settle_s = float(self.declare_parameter("perception_settle_s", 2.5).value)

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
            make_provider(provider_name, model=model, requests_per_minute=rpm,
                          log=lambda m: self.get_logger().warn(m), thinking_level=thinking,
                          timeout_s=llm_timeout), registry, SYSTEM_PROMPT,
            max_steps=self.max_steps,
            on_event=self._log_event,
        )
        self.get_logger().info(f"Reasoning layer up (provider={provider_name}, model={model}, "
                               f"thinking={thinking or 'default'}).")

    def _log_event(self, e):
        if e["phase"] == "llm":
            self.get_logger().info(f"[step {e['step']}] asking the LLM for the next action...")
        elif e["phase"] == "start":
            thought = f" | thought: {e['thought']}" if e["thought"] else ""
            self.get_logger().info(f"[step {e['step']}] calling {e['tool']}({e['arguments']}){thought}")
        else:
            self.get_logger().info(f"[step {e['step']}] {e['tool']} -> ok={e['ok']}: {e['result']}")

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
        p, q = msg.current_pose.position, msg.current_pose.orientation
        self.telemetry = {
            "armed": msg.armed, "px4_nav_state": msg.px4_nav_state,
            "controller_state": msg.controller_state, "in_air": msg.is_in_air,
            "position": [round(p.x, 2), round(p.y, 2), round(p.z, 2)],
            "heading_deg": round(math.degrees(2.0 * math.atan2(q.z, q.w)) % 360.0, 1),
            "battery_percent": None if math.isnan(msg.battery_percentage) else round(msg.battery_percentage, 1),
            "geofence_ok": msg.geofence_ok,
            "last_event": msg.last_event,
        }

    # ---- tools ----------------------------------------------------------

    def _register_tools(self, r: ToolRegistry):
        r.register("get_telemetry", "Current vehicle state, position and heading (map frame), battery.",
                   obj({}), self._tool_telemetry)
        r.register("get_scene_graph", "Objects currently tracked by perception, with relations.",
                   obj({}), self._tool_scene)
        r.register("takeoff", "Arm and take off to an altitude in metres; blocks until holding there. "
                   "The result includes what perception sees on arrival (scene).",
                   obj({"altitude": NUM}, ["altitude"]), self._tool_takeoff)
        r.register("navigate_to",
                   "Fly to (x, y, z) in the map frame; blocks until reached or failed. Optional yaw_deg "
                   "sets the heading on arrival (map frame: 0 = east/+x, 90 = north/+y); the camera looks "
                   "along the heading. Omit it to keep the current heading. The result includes what "
                   "perception sees on arrival (scene).",
                   obj({"x": NUM, "y": NUM, "z": NUM, "yaw_deg": NUM}, ["x", "y", "z"]), self._tool_navigate)
        r.register("scan",
                   "Turn a full circle in place in steps of step_deg (default 60) and return every object "
                   "perception reported at any heading, with its position. One call replaces a series of "
                   "turns and get_scene_graph calls. Requires being airborne.",
                   obj({"step_deg": NUM}), self._tool_scan)
        r.register("hold", "Stop and hold the current position.", obj({}), lambda: self._flight("HOLD"))
        r.register("land", "Land at the current position; blocks until landed and disarmed.",
                   obj({}), lambda: self._tool_land("LAND", "Landing"))
        r.register("return_to_launch", "Fly home and land; blocks until landed and disarmed.",
                   obj({}), lambda: self._tool_land("RTL", "Return to launch"))
        r.register("query_memory", "Search procedures and past mission reports.",
                   obj({"query": {"type": "string"}}, ["query"]), self._tool_memory)

    def _tool_telemetry(self):
        if not self.telemetry:
            raise RuntimeError("No telemetry received from the action layer.")
        return dict(self.telemetry)

    def _battery_gate(self, minimum: float, action: str):
        battery = self.telemetry.get("battery_percent")
        if battery is not None and battery < minimum:
            raise RuntimeError(f"Battery {battery:.0f}% is below {minimum:.0f}% required to {action}; "
                               f"return to launch or land instead.")

    def _tool_takeoff(self, altitude: float):
        self._battery_gate(MIN_BATTERY_TAKEOFF, "take off")
        self._flight("TAKEOFF", altitude)
        t = wait_takeoff(self._tool_telemetry_or_none, float(altitude), self.nav_timeout)
        return {"message": f"Holding at {t['position'][2]:.1f} m.", "position": t["position"],
                "scene": self._settled_scene()}

    def _tool_land(self, command: str, what: str):
        self._flight(command)
        t = wait_landed(self._tool_telemetry_or_none, self.nav_timeout, what=what)
        return {"message": f"{what} complete: landed and disarmed.", "position": t["position"]}

    def _tool_telemetry_or_none(self):
        return dict(self.telemetry) if self.telemetry else None

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

    def _settled_scene(self):
        """Scene graph nodes after perception has had settle_s to process frames at the new pose."""
        time.sleep(self.settle_s)
        if self.scene["stamp"] is None:
            return "No scene graph received from the perception layer."
        return compact_nodes(self.scene["nodes"])

    def _tool_navigate(self, x: float, y: float, z: float, yaw_deg: float = None):
        result = self._navigate(x, y, z, yaw_deg)
        result["scene"] = self._settled_scene()
        return result

    def _tool_scan(self, step_deg: float = 60.0):
        t = self._tool_telemetry()
        if not t["in_air"] or t["controller_state"] not in ("HOLDING", "NAVIGATING"):
            raise RuntimeError(f"Scan needs the vehicle airborne under control (state {t['controller_state']}).")
        x, y, z = t["position"]
        seen = {}
        headings = scan_headings(t["heading_deg"], step_deg)
        for i, h in enumerate([t["heading_deg"]] + headings):
            if i:
                self._navigate(x, y, z, h)
            nodes = self._settled_scene()
            if not isinstance(nodes, list):
                raise RuntimeError(nodes)
            merge_sightings(seen, nodes, h)
        objects = sorted(seen.values(), key=lambda n: (not n["confirmed"], -n["existence_prob"]))
        return {"position": [x, y, z], "headings_deg": [t["heading_deg"]] + headings, "objects": objects}

    def _navigate(self, x: float, y: float, z: float, yaw_deg: float = None):
        self._battery_gate(MIN_BATTERY_NAVIGATE, "navigate")
        if not self.nav_client.wait_for_server(timeout_sec=self.service_timeout):
            raise RuntimeError("Action layer navigate_to server unavailable.")
        goal = NavigateTo.Goal()
        goal.target.x, goal.target.y, goal.target.z = float(x), float(y), float(z)
        goal.yaw = math.nan if yaw_deg is None else math.radians(float(yaw_deg))
        handle = wait_future(self.nav_client.send_goal_async(goal), self.service_timeout)
        if not handle.accepted:
            raise RuntimeError("Navigation goal rejected.")
        res = wait_future(handle.get_result_async(), self.nav_timeout).result
        if not res.success:
            raise RuntimeError(res.message)
        p = res.final_position
        return {"message": res.message, "final_position": [round(p.x, 2), round(p.y, 2), round(p.z, 2)]}

    def _tool_memory(self, query: str):
        # The learning layer is optional (off on small VMs), so don't wait for it.
        if not self.rag_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError("Learning layer rag_query service unavailable (memory is not running).")
        res = wait_future(self.rag_client.call_async(
            RAGQuery.Request(query_text=query, top_k=3, filter_category="")), self.service_timeout)
        if not res.success:
            raise RuntimeError(res.error_message)
        return [{"id": i, "score": round(s, 3), "content": c}
                for i, c, s in zip(res.document_ids, res.contents, res.similarity_scores)]

    # ---- mission service ------------------------------------------------

    def _save_trace(self, goal: str, result):
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            path = os.path.join(self.trace_dir, time.strftime("mission_%Y%m%d_%H%M%S.json"))
            with open(path, "w") as f:
                json.dump({"goal": goal, "success": result.success, "summary": result.summary,
                           "trace": result.trace}, f, indent=2, default=str)
            self.get_logger().info(f"Trace saved to {path}")
        except OSError as ex:
            self.get_logger().warn(f"Could not save mission trace: {ex}")

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
            self._save_trace(request.goal, result)
        except Exception as ex:
            response.success, response.summary = False, f"Mission error: {ex}"
            self.get_logger().error(response.summary)
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
