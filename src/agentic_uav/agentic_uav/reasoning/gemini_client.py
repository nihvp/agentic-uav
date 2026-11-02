"""
Gemini API Client for UAV Cognitive Engine.
Supports the Google GenAI SDK with graceful offline cognitive fallback
for air-gapped simulation benches and automated CI testing.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("agentic_uav.reasoning.gemini")


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.client = None
        self.is_offline_mode = False

        self._initialize_client()

    def _initialize_client(self):
        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Switching to Deterministic ReAct Simulation Mode. "
                "Set GEMINI_API_KEY to enable live multimodal cloud reasoning."
            )
            self.is_offline_mode = True
            return

        try:
            # Try new google-genai SDK
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.sdk_type = "genai"
            logger.info(f"Initialized Google GenAI SDK with model {self.model_name}")
        except ImportError:
            try:
                # Try google-generativeai legacy package
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                self.client = genai_legacy.GenerativeModel(self.model_name)
                self.sdk_type = "legacy"
                logger.info(f"Initialized google.generativeai legacy with model {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load Gemini SDK ({e}). Operating in offline ReAct mode.")
                self.is_offline_mode = True

    def generate_response(self, system_instruction: str, prompt: str) -> str:
        """Invokes Gemini API or generates deterministic ReAct output."""
        if not self.is_offline_mode and self.client is not None:
            try:
                if self.sdk_type == "genai":
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={"system_instruction": system_instruction, "temperature": 0.2}
                    )
                    return response.text
                elif self.sdk_type == "legacy":
                    full_prompt = f"System: {system_instruction}\nUser: {prompt}"
                    response = self.client.generate_content(full_prompt)
                    return response.text
            except Exception as ex:
                logger.error(f"Gemini API invocation failed: {ex}. Using fallback planner.")

        # Deterministic cognitive ReAct engine for simulation & offline verification
        return self._generate_simulated_react(prompt)

    def _generate_simulated_react(self, prompt: str) -> str:
        """Dynamic rule-guided ReAct simulator that parses actual entities and coordinates from the scene graph."""
        # Parse nodes from prompt:
        # Format: - [ID 1] survivor at (8.0, 4.0, 0.0) (P_exist=0.95) attrs:[...]
        node_pattern = re.compile(
            r'-\s*\[ID\s*(\d+)\]\s*([a-zA-Z0-9_-]+)\s*at\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)',
            re.IGNORECASE
        )
        survivors = []
        fires = []
        for match in node_pattern.finditer(prompt):
            nid, label, x, y, z = match.groups()
            coords = [float(x), float(y), float(z)]
            label_lower = label.lower()
            if "survivor" in label_lower or "person" in label_lower:
                # Check if survivor is already safe at the emergency landing pad (near x=0, y=8)
                dist_pad = ((coords[0] - 0.0)**2 + (coords[1] - 8.0)**2)**0.5
                if dist_pad > 2.2:  # Not yet rescued!
                    survivors.append({"id": int(nid), "pos": coords})
            elif "fire" in label_lower or "hazard" in label_lower:
                fires.append({"id": int(nid), "pos": coords})

        # Find the survivor facing the highest wildfire threat
        target_survivor = None
        target_fire = None
        min_threat_dist = float("inf")

        for s in survivors:
            for f in fires:
                d = ((s["pos"][0] - f["pos"][0])**2 + (s["pos"][1] - f["pos"][1])**2)**0.5
                if d < min_threat_dist:
                    min_threat_dist = d
                    target_survivor = s
                    target_fire = f

        if target_survivor is None and survivors:
            target_survivor = survivors[0]

        # Case 1: Survivor in proximity to active wildfire
        if target_survivor is not None and min_threat_dist < 15.0 and target_fire is not None:
            sx, sy = target_survivor["pos"][0], target_survivor["pos"][1]
            fx, fy = target_fire["pos"][0], target_fire["pos"][1]
            return (
                f"Thought: 3D scene graph indicates survivor at ({sx:.1f}, {sy:.1f}) in critical proximity to fire hazard at ({fx:.1f}, {fy:.1f}) (distance: {min_threat_dist:.1f}m). "
                f"Immediate safety protocols require isolating the hazard vector, dispatching an alert via MCP, and commanding UAV aerial coverage to ({sx:.1f}, {sy:.1f}, 4.0).\n"
                f"Action: invoke_mcp_tool\n"
                f"Action Input: {{\"tool_name\": \"emergency_alert_dispatch\", \"arguments\": {{\"severity\": \"CRITICAL\", \"threat\": \"Wildfire near survivor\", \"location\": [{sx:.1f}, {sy:.1f}, 0.0]}}}}\n"
            )

        # Case 2: Survivor detected without direct fire threat
        if target_survivor is not None:
            sx, sy = target_survivor["pos"][0], target_survivor["pos"][1]
            return (
                f"Thought: Survivor detected at ({sx:.1f}, {sy:.1f}) in scene graph. Commanding UAV to navigate and inspect target coordinates.\n"
                f"Action: inspect_target\n"
                f"Action Input: {{\"target_id\": {target_survivor['id']}, \"duration\": 5.0, \"x\": {sx:.1f}, \"y\": {sy:.1f}, \"z\": 4.0}}\n"
            )

        # Case 3: Landing zone blocked by debris
        prompt_lower = prompt.lower()
        if "landing_zone" in prompt_lower and "debris" in prompt_lower:
            return (
                "Thought: The designated landing zone has a debris obstacle relation. "
                "I will query the RAG memory for secondary landing site alternatives and obstacle clearance SOPs.\n"
                "Action: query_rag_memory\n"
                "Action Input: {\"query\": \"Standard procedure for obstructed emergency landing zone\"}\n"
            )

        # Case 4: All survivors rescued / area secure -> Return to Launch
        return (
            "Thought: All detected survivors are secured at the landing pad. Search objectives complete. Commanding Return-to-Launch.\n"
            "Action: return_to_launch\n"
            "Action Input: {\"reason\": \"All survivors secured at landing pad\"}\n"
        )
