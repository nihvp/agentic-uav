#!/usr/bin/env bash
# =============================================================================
# Reset Drone Pose and Fly in Plain Sight
# =============================================================================
set -u

echo "--- 1. Halting any current velocity ---"
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 0.0}" 2>/dev/null || true

echo "--- 2. Teleporting drone back to (0.0, 0.0, 2.0) right in front of camera ---"
gz service -s /world/search_and_rescue_harmonic/set_pose \
  --reqtype gz.msgs.Pose \
  --reptype gz.msgs.Boolean \
  --timeout 2000 \
  --req 'name: "agentic_uav", position: {x: 0.0, y: 0.0, z: 2.0}, orientation: {w: 1.0}' 2>/dev/null || true

echo "Look at your Gazebo window — the blue drone should now be floating at 2m height."
echo ""
echo "--- 3. Checking live drone coordinates from Gazebo odometry ---"
timeout 2 gz topic -e -t /uav/odometry -n 1 2>/dev/null | grep -A 4 "position" || echo "Odometry sample pending..."
echo ""

echo "--- 4. Moving drone forward in +X direction across your screen ---"
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 1.5, y: 0.0, z: 0.0}" 2>/dev/null
sleep 2
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 0.0}" 2>/dev/null
echo "Done! The drone moved forward by 3 meters and stopped."
