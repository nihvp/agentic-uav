#!/usr/bin/env bash
# =============================================================================
# UAV Bridge Diagnostic Script
# Run inside the container:  docker exec -it agentic_uav_container bash /ros2_ws/src/diagnose.sh
# =============================================================================
set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILURES=$((FAILURES+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
FAILURES=0

echo "============================================="
echo "  Agentic UAV Bridge Diagnostic"
echo "============================================="
echo ""

# 1. Check ROS nodes are running
echo "--- Step 1: ROS 2 Nodes ---"
NODES=$(ros2 node list 2>/dev/null)
echo "$NODES"
echo "$NODES" | grep -q "/ros_gz_bridge" && pass "ros_gz_bridge node is running" || fail "ros_gz_bridge node NOT found"
echo "$NODES" | grep -q "/action_node" && pass "action_node is running" || fail "action_node NOT found"
echo ""

# 2. Check ROS topics
echo "--- Step 2: ROS 2 Topics ---"
TOPICS=$(ros2 topic list 2>/dev/null)
echo "$TOPICS" | grep -q "/cmd_vel" && pass "ROS topic /cmd_vel exists" || fail "ROS topic /cmd_vel NOT found"
echo "$TOPICS" | grep -q "/uav/odometry" && pass "ROS topic /uav/odometry exists" || fail "ROS topic /uav/odometry NOT found"
echo "$TOPICS" | grep -q "/action/agent_command" && pass "ROS topic /action/agent_command exists" || fail "ROS topic /action/agent_command NOT found"
echo ""

# 3. Check what Gazebo topics exist
echo "--- Step 3: Gazebo Transport Topics ---"
GZ_TOPICS=$(gz topic -l 2>/dev/null)
echo "$GZ_TOPICS" | grep -q "/cmd_vel" && pass "GZ topic /cmd_vel exists" || warn "GZ topic /cmd_vel not found in gz topic list"
echo "$GZ_TOPICS" | grep -q "/model/agentic_uav/cmd_vel" && warn "GZ topic /model/agentic_uav/cmd_vel also exists (possible old VelocityControl)" || pass "No stale /model/agentic_uav/cmd_vel topic"
echo "$GZ_TOPICS" | grep -q "/uav/odometry" && pass "GZ topic /uav/odometry exists" || fail "GZ topic /uav/odometry NOT found"
echo ""

# 4. Check if odometry is actually flowing
echo "--- Step 4: Odometry Flow Check ---"
ODOM_MSG=$(timeout 2 ros2 topic echo --once /uav/odometry 2>/dev/null | head -6 || true)
if [ -n "$ODOM_MSG" ]; then
    pass "Odometry is flowing on ROS /uav/odometry"
    echo "$ODOM_MSG"
else
    # Check GZ side
    GZ_ODOM=$(timeout 2 gz topic -e -t /uav/odometry -n 1 2>/dev/null | head -6 || true)
    if [ -n "$GZ_ODOM" ]; then
        warn "GZ publishes /uav/odometry, but ROS bridge has not forwarded it"
    else
        warn "Gazebo has not published to /uav/odometry yet"
    fi
fi
echo ""

# 5. Test sending a velocity command via Gazebo directly
echo "--- Step 5: Direct Gazebo Velocity Test ---"
echo "Sending gz topic command to lift drone (z=2.0 for 1 second)..."
gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 2.0}" 2>/dev/null
sleep 1
gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 0.0}" 2>/dev/null
echo "Check Gazebo window - did the drone lift off?"
echo "If YES: VelocityControl plugin works on /cmd_vel"
echo "If NO:  VelocityControl plugin might be listening on /model/agentic_uav/cmd_vel instead"
echo ""

# 5b. Also try the old topic
echo "Also trying the old topic /model/agentic_uav/cmd_vel..."
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 2.0}" 2>/dev/null
sleep 1
gz topic -t "/model/agentic_uav/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.0, y: 0.0, z: 0.0}" 2>/dev/null
echo "If the drone moved on THIS command but not the previous one,"
echo "then the SDF file was NOT reloaded (old container still running)."
echo ""

# 6. Test ROS->Gazebo bridge
echo "--- Step 6: ROS->GZ Bridge Test ---"
echo "Publishing momentary Twist via ROS to /cmd_vel..."
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 1.0}}" 2>/dev/null || true
sleep 1
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}}" 2>/dev/null || true
echo "Drone stopped."
echo ""

# 7. Summary
echo "============================================="
echo "  Diagnostic Summary"
echo "============================================="
if [ "$FAILURES" -eq 0 ]; then
    pass "All checks passed"
else
    fail "$FAILURES check(s) failed - see above"
fi
echo ""
echo "IMPORTANT: If the drone moved with 'gz topic' in Step 5 but NOT"
echo "with 'ros2 topic pub' in Step 6, the bridge is not forwarding."
echo "Run:  ros2 topic info /cmd_vel -v"
echo "to see who is publishing and subscribing."
