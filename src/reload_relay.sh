#!/usr/bin/env bash
pkill -f "ros_to_gz_relay.py" 2>/dev/null || true
sleep 0.5
python3 /ros2_ws/src/ros_to_gz_relay.py > /tmp/relay.log 2>&1 &
sleep 1.0
echo "=== Relay Reloaded: Now Supporting All AI Actions (INSPECT_TARGET, WAYPOINTS, RTL) ==="
