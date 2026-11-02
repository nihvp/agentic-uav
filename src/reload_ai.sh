#!/usr/bin/env bash
# =============================================================================
# Reload Perception Layer, Reasoning Layer & Relay
# =============================================================================
pkill -f "perception_node" 2>/dev/null || true
pkill -f "reasoning_node" 2>/dev/null || true
pkill -f "ros_to_gz_relay.py" 2>/dev/null || true
sleep 1.0

PYTHONPATH=/ros2_ws/src/agentic_uav:$PYTHONPATH python3 /ros2_ws/src/agentic_uav/scripts/perception_node > /tmp/perception.log 2>&1 &
PYTHONPATH=/ros2_ws/src/agentic_uav:$PYTHONPATH python3 /ros2_ws/src/agentic_uav/scripts/reasoning_node > /tmp/reasoning.log 2>&1 &
python3 /ros2_ws/src/ros_to_gz_relay.py > /tmp/relay.log 2>&1 &
sleep 2.0

echo "=== Perception, Reasoning & Flight Relay Reloaded & Live! ==="
