"""
Model Context Protocol (MCP) Tool Registry and Security Orchestrator.
Manages tool schema exposure, argument schema validation, secure token verification,
and external service integrations (Emergency Dispatch, Airspace UTM, Weather, Satellite).
"""

import json
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger("agentic_uav.integration.mcp")


class MCPToolRegistry:
    """Implements Model Context Protocol (MCP) tools schema and dispatch engine."""

    def __init__(self, allowed_tokens: Optional[List[str]] = None):
        self.allowed_tokens = set(allowed_tokens or ["uav_reasoning_node_auth_token", "admin_secret_token"])
        self.tools = {
            "emergency_alert_dispatch": {
                "name": "emergency_alert_dispatch",
                "description": "Dispatches high-priority incident reports to regional Emergency Operations Center (EOC) CAD systems.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["LOW", "MODERATE", "CRITICAL"]},
                        "threat": {"type": "string", "description": "Description of hazard or victim status"},
                        "location": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z] coordinates"}
                    },
                    "required": ["severity", "threat", "location"]
                },
                "handler": self._handle_emergency_dispatch
            },
            "utm_airspace_clearance": {
                "name": "utm_airspace_clearance",
                "description": "Requests dynamic UTM (Unmanned Traffic Management) airspace authorization and collision avoidance clearance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "corridor_radius": {"type": "number"},
                        "altitude_ceiling": {"type": "number"}
                    },
                    "required": ["altitude_ceiling"]
                },
                "handler": self._handle_utm_clearance
            },
            "weather_advisory_feed": {
                "name": "weather_advisory_feed",
                "description": "Fetches micro-meteorological observations (wind gusts, precipitation, visibility) for the UAV flight zone.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "zone_id": {"type": "string"}
                    }
                },
                "handler": self._handle_weather_feed
            },
            "satellite_imagery_query": {
                "name": "satellite_imagery_query",
                "description": "Retrieves pre-disaster high-resolution satellite imagery tiles for structural comparison.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bounding_box": {"type": "array", "items": {"type": "number"}}
                    }
                },
                "handler": self._handle_satellite_query
            }
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns standard MCP tools catalog."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"]
            }
            for t in self.tools.values()
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], client_token: str) -> Tuple[bool, str, str]:
        """
        Validates client authentication token and schema before orchestrating tool execution.
        Returns: (success: bool, result_json: str, error_message: str)
        """
        # Security Verification
        if client_token not in self.allowed_tokens:
            logger.warning(f"Unauthorized MCP invocation attempt with token '{client_token[:6]}...'")
            return False, "", "HTTP 401 Unauthorized: Invalid or expired MCP client token."

        if tool_name not in self.tools:
            return False, "", f"Tool '{tool_name}' not found in MCP registry."

        tool_spec = self.tools[tool_name]

        # Parameter schema verification
        req_params = tool_spec["inputSchema"].get("required", [])
        for p in req_params:
            if p not in arguments:
                return False, "", f"Missing required parameter '{p}' for tool '{tool_name}'."

        try:
            handler = tool_spec["handler"]
            result = handler(arguments)
            return True, json.dumps(result), ""
        except Exception as e:
            logger.error(f"Error executing MCP tool '{tool_name}': {e}")
            return False, "", f"Execution failure: {str(e)}"

    def _handle_emergency_dispatch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        severity = args.get("severity", "CRITICAL")
        threat = args.get("threat", "Unknown threat")
        loc = args.get("location", [0, 0, 0])
        logger.info(f"[EOC DISPATCH] Incident recorded: {threat} at {loc} with Severity={severity}")
        
        # Schedule survivor evacuation to safe landing pad upon first responder arrival
        try:
            import subprocess
            subprocess.Popen(
                ["gz", "service", "-s", "/world/search_and_rescue_harmonic/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean", "--timeout", "1000",
                 "--req", 'name: "survivor_target", position: {x: 0.0, y: 8.0, z: 0.0}, orientation: {w: 1.0}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info("[SIMULATION] Ground team dispatched: moving survivor to Emergency Landing Pad (0, 8, 0).")
        except Exception:
            pass

        return {
            "dispatch_status": "ACKNOWLEDGED",
            "incident_ticket_id": "CAD-UAV-9942",
            "assigned_ground_unit": "Rescue Squad 4",
            "estimated_arrival_minutes": 8.5
        }

    def _handle_utm_clearance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ceiling = args.get("altitude_ceiling", 25.0)
        return {
            "authorization": "GRANTED",
            "transponder_code": "4421-SITL",
            "active_conflicts": 0,
            "airspace_status": "CORRIDOR_RESERVED",
            "max_allowed_altitude_msl": float(ceiling)
        }

    def _handle_weather_feed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "station": "KPAO-SIM",
            "wind_speed_mps": 2.8,
            "wind_direction_deg": 135,
            "gust_speed_mps": 4.1,
            "visibility_km": 10.0,
            "flight_condition": "VFR_OPTIMAL",
            "precipitation": "NONE"
        }

    def _handle_satellite_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "provider": "Copernicus-Sentinel-2",
            "capture_timestamp": "2026-10-28T14:22:00Z",
            "resolution_m_per_pixel": 0.5,
            "status": "IMAGERY_STREAM_SYNCED"
        }
