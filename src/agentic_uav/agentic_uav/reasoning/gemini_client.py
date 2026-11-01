"""
Gemini API Client for UAV Cognitive Engine.
Supports the Google GenAI SDK with graceful offline cognitive fallback
for air-gapped simulation benches and automated CI testing.
"""

import os
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
        """High-level rule-guided ReAct simulator mimicking Gemini's structured output."""
        prompt_lower = prompt.lower()

        if "fire_hazard" in prompt_lower and "survivor" in prompt_lower:
            return (
                "Thought: The probabilistic scene graph indicates a fire hazard threatening a survivor nearby. "
                "Immediate safety protocols require isolating the hazard vector, dispatching an alert, and verifying a clear landing route.\n"
                "Action: invoke_mcp_tool\n"
                "Action Input: {\"tool_name\": \"emergency_alert_dispatch\", \"arguments\": {\"severity\": \"CRITICAL\", \"threat\": \"Wildfire near survivor\", \"location\": [12.4, 5.2, 0.0]}}\n"
            )
        elif "survivor" in prompt_lower and "inspect" not in prompt_lower:
            return (
                "Thought: A potential survivor has been detected in the scene graph with high existence probability. "
                "I will command the UAV to loiter directly above the coordinates to perform target inspection and confirm vital signs.\n"
                "Action: inspect_target\n"
                "Action Input: {\"target_id\": 1, \"duration\": 10.0, \"x\": 10.5, \"y\": 4.0, \"z\": 5.0}\n"
            )
        elif "landing_zone" in prompt_lower and "debris" in prompt_lower:
            return (
                "Thought: The designated landing zone has a debris obstacle relation. "
                "I should query the RAG memory for secondary landing site alternatives and obstacle clearance SOPs.\n"
                "Action: query_rag_memory\n"
                "Action Input: {\"query\": \"Standard procedure for obstructed emergency landing zone\"}\n"
            )
        else:
            return (
                "Thought: Survey sector Alpha is clear of active hazards. Advancing along the search grid pattern.\n"
                "Action: navigate_waypoint\n"
                "Action Input: {\"x\": 15.0, \"y\": 15.0, \"z\": 6.0, \"speed\": 2.5}\n"
            )
