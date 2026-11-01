#!/usr/bin/env bash
set -e

# Source ROS 2 Humble base setup
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source workspace install setup
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source "/ros2_ws/install/setup.bash"
fi

# Set Gazebo Harmonic resource path
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}:/ros2_ws/src/agentic_uav/worlds"
export PYTHONUNBUFFERED=1

# Execute passed command
exec "$@"
