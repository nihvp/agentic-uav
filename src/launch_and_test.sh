#!/usr/bin/env bash
# =============================================================================
# Restart Relay with Non-blocking Popen & Fly to (8.0, 4.0, 4.0)
# =============================================================================
set -e

echo "=== [1/2] Restarting High-Speed Relay ==="
pkill -f "ros_to_gz_relay.py" 2>/dev/null || true
sleep 0.5
python3 /ros2_ws/src/ros_to_gz_relay.py > /tmp/relay.log 2>&1 &
sleep 1.5

echo "=== [2/2] Publishing Autonomous Waypoint: (8.0, 4.0, 4.0) at 2.0 m/s ==="
ros2 topic pub --once /action/agent_command agentic_uav_interfaces/msg/AgentCommand "{
  action_type: 'NAVIGATE_WAYPOINT',
  target_position: {x: 8.0, y: 4.0, z: 4.0},
  speed: 2.0
}"

echo ""
echo "Watch your Gazebo window right now — the drone is physically cruising to (8, 4, 4)!"
