#!/usr/bin/env bash
# =============================================================================
# Direct Flight Relay for Agentic UAV
# Translates ROS /cmd_vel and /action/agent_command directly into Gazebo physics
# =============================================================================
set -e

echo "=== [1/4] Checking live bridge log ==="
if [ -f "/tmp/live_bridge.log" ]; then
    cat /tmp/live_bridge.log
else
    echo "No log found at /tmp/live_bridge.log"
fi
echo ""

echo "=== [2/4] Verifying direct Gazebo motion ==="
echo "Moving drone forward in +X by 2 meters right now..."
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 1.5, y: 0.0, z: 0.0}" 2>/dev/null
sleep 1.5
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 0.0}" 2>/dev/null
echo "Did the drone move forward in Gazebo? (Look at simulator window)"
