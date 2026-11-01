"""
ReAct (Reasoning and Acting) Engine for UAV autonomy.
Orchestrates the cognitive cycle:
Thought -> Action -> Action Input -> (Action Execution) -> Observation -> Next Thought.
"""

import json
import re
from typing import Dict, Any, Tuple, Optional, List
from agentic_uav.reasoning.gemini_client import GeminiClient

REACT_SYSTEM_PROMPT = """
You are the Cognitive Reasoning Engine for an autonomous search-and-rescue UAV operating under ROS 2 and PX4.
You receive a real-time 3D Probabilistic Semantic Scene Graph, vehicle flight telemetry, and observations from previous actions.
You must solve the mission goal by executing a ReAct (Reasoning + Acting) loop.

Use the following format strictly:

Thought: [Your chain-of-thought analysis of the 3D scene graph, threats, spatial geometry, and flight telemetry]
Action: [The tool name to execute, exactly one from the Allowed Tools list]
Action Input: [A valid JSON object containing the tool parameters]

Allowed Tools:
1. navigate_waypoint: Fly the UAV to target coordinates.
   Parameters: {"x": float, "y": float, "z": float, "speed": float}
2. inspect_target: Position UAV above a detected entity to gather high-res inspection data.
   Parameters: {"target_id": int, "duration": float, "x": float, "y": float, "z": float}
3. query_rag_memory: Search cross-mission memory and SOPs for protocols or lessons learned.
   Parameters: {"query": str}
4. invoke_mcp_tool: Call external secured tool via Model Context Protocol (e.g. weather, UTM clearance, dispatch).
   Parameters: {"tool_name": str, "arguments": object}
5. deploy_payload: Drop first-aid kit or rescue beacon over a target coordinate.
   Parameters: {"target_id": int, "payload_type": str}
6. return_to_launch: Abort mission and return drone home due to hazard, low battery, or mission complete.
   Parameters: {"reason": str}
7. mission_complete: Conclude mission when all objectives are confirmed.
   Parameters: {"summary": str}
"""


class ReActEngine:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client
        self.history: List[Dict[str, str]] = []

    def format_state_prompt(self, mission_goal: str,
                            scene_nodes: List[Dict[str, Any]],
                            scene_edges: List[Dict[str, Any]],
                            telemetry: Dict[str, Any],
                            last_observation: Optional[str] = None) -> str:
        """Constructs a comprehensive contextual prompt incorporating the 3D Scene Graph."""
        nodes_summary = []
        for n in scene_nodes:
            pos_str = f"({n.get('position', [0,0,0])[0]:.1f}, {n.get('position', [0,0,0])[1]:.1f}, {n.get('position', [0,0,0])[2]:.1f})"
            prob = n.get('existence_prob', 1.0)
            label = n.get('label', 'unknown')
            attrs = ",".join(n.get('attributes', []))
            nodes_summary.append(f" - [ID {n.get('id')}] {label} at {pos_str} (P_exist={prob:.2f}) attrs:[{attrs}]")

        edges_summary = []
        for e in scene_edges:
            edges_summary.append(f" - (ID {e.get('source_id')}) --[{e.get('relation')} P={e.get('probability', 1.0):.2f}]--> (ID {e.get('target_id')}) dist={e.get('distance', 0.0):.1f}m")

        telem_str = (
            f"Armed: {telemetry.get('armed', False)}, Mode: {telemetry.get('mode', 'OFFBOARD')}, "
            f"Battery: {telemetry.get('battery', 100.0):.1f}%, Alt: {telemetry.get('altitude', 0.0):.1f}m, "
            f"Geofence: {telemetry.get('geofence_ok', True)}"
        )

        prompt = (
            f"CURRENT MISSION GOAL: {mission_goal}\n\n"
            f"FLIGHT TELEMETRY: {telem_str}\n\n"
            f"3D SEMANTIC SCENE GRAPH:\n"
            f"Nodes ({len(nodes_summary)} entities):\n" + ("\n".join(nodes_summary) if nodes_summary else " - No objects currently tracked.\n") +
            f"\nRelational Edges:\n" + ("\n".join(edges_summary) if edges_summary else " - None.\n")
        )

        if last_observation:
            prompt += f"\nOBSERVATION FROM PREVIOUS ACTION:\n{last_observation}\n"

        prompt += "\nProvide the next Thought, Action, and Action Input:"
        return prompt

    def step(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """Executes one ReAct inference step, parsing Thought, Action, and Action Input."""
        raw_output = self.gemini.generate_response(
            system_instruction=REACT_SYSTEM_PROMPT,
            prompt=prompt
        )

        thought, action, action_input = self._parse_react_output(raw_output)
        return thought, action, action_input

    def _parse_react_output(self, text: str) -> Tuple[str, str, Dict[str, Any]]:
        thought = "Analyzing environmental state."
        action = "navigate_waypoint"
        action_input = {"x": 5.0, "y": 5.0, "z": 4.0, "speed": 2.0}

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                # Basic key-value recovery
                action_input = {"raw": input_match.group(1).strip()}
        else:
            # Check if there is an unbracketed string or alternate format
            input_line_match = re.search(r"Action Input:\s*(.+)", text)
            if input_line_match:
                content = input_line_match.group(1).strip()
                try:
                    action_input = json.loads(content)
                except Exception:
                    action_input = {"query": content}

        return thought, action, action_input
