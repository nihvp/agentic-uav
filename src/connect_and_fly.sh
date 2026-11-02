#!/usr/bin/env bash
# =============================================================================
# Connect Bridge and Fly Script
# Run inside container: bash /ros2_ws/src/connect_and_fly.sh
# =============================================================================
set -e

echo "=== [1/3] Halting any lingering upward drift in Gazebo ==="
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 0.0}" 2>/dev/null || true

echo "=== [2/3] Starting Live ROS-GZ Bridge for /model/agentic_uav/cmd_vel & /uav/odometry ==="
# Launch dedicated bridge node in background if not already running
if ! ros2 node list 2>/dev/null | grep -q "live_bridge"; then
    ros2 run ros_gz_bridge parameter_bridge \
        /model/agentic_uav/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
        /uav/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
        --ros-args -r __node:=live_bridge > /tmp/live_bridge.log 2>&1 &
    sleep 2
    echo "Bridge is online and connected!"
else
    echo "Bridge is already running."
fi

echo "=== [3/3] Sending Autonomous Waypoint Command: (8.0, 4.0, 4.0) at 2.0 m/s ==="
ros2 topic pub --once /action/agent_command agentic_uav_interfaces/msg/AgentCommand "{
  action_type: 'NAVIGATE_WAYPOINT',
  target_position: {x: 8.0, y: 4.0, z: 4.0},
  speed: 2.0
}"

echo ""
echo "Command dispatched! Look at your Gazebo Sim 3D window — the drone is flying to (8, 4, 4)."
