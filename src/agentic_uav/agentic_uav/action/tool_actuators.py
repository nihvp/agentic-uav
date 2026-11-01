"""
Atomic Digital and Physical Tool Actuator Dispatcher.
Executes non-flight tool invocations such as camera gimbal re-targeting,
sensor mode activation, beacon strobes, and payload release solenoids.
"""

import json
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("agentic_uav.action.actuators")


class ToolActuators:
    def __init__(self):
        self.gimbal_pitch = -45.0  # degrees (nadir is -90)
        self.gimbal_yaw = 0.0
        self.thermal_active = False
        self.payload_released = False
        self.strobe_active = False

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Dispatches tool command to hardware simulation interface."""
        name = tool_name.lower()

        if "gimbal" in name:
            target_id = params.get("target_id", 0)
            pitch = params.get("pitch", -60.0)
            self.gimbal_pitch = float(pitch)
            logger.info(f"Gimbal articulated towards target {target_id} at pitch {pitch} deg.")
            return True, f"Gimbal oriented to pitch {pitch} deg."

        elif "thermal" in name:
            self.thermal_active = params.get("enable", True)
            state_str = "ENABLED" if self.thermal_active else "STANDBY"
            logger.info(f"Thermal sensor switched to {state_str}.")
            return True, f"Thermal sensor set to {state_str}"

        elif "payload" in name or "release" in name or "drop" in name:
            if self.payload_released:
                return False, "Payload already deployed (magazine empty)."
            payload_type = params.get("payload_type", "Emergency First-Aid Kit")
            self.payload_released = True
            logger.info(f"Payload actuator triggered. Released: {payload_type}")
            return True, f"Successfully deployed {payload_type} at current coordinates."

        elif "strobe" in name or "beacon" in name:
            self.strobe_active = not self.strobe_active
            status = "ON" if self.strobe_active else "OFF"
            logger.info(f"Emergency locator strobe: {status}")
            return True, f"Emergency strobe turned {status}"

        else:
            return False, f"Unknown tool actuator '{tool_name}'."
