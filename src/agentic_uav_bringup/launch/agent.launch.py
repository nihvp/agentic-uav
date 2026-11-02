"""Launch the agent side: camera bridge + the five layers.

PX4 SITL and Gazebo run in the `sim` container; this connects to them over
uXRCE-DDS (PX4 topics) and gz-transport (camera bridge).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("agentic_uav_bringup")
    params = os.path.join(share, "config", "params.yaml")
    bridge = os.path.join(share, "config", "gz_bridge.yaml")

    layers = ["action", "perception", "reasoning", "learning"]
    args = [DeclareLaunchArgument(f"{name}", default_value="true") for name in layers]
    args.append(DeclareLaunchArgument("bridge", default_value="true"))

    def layer(name):
        return Node(
            package="agentic_uav", executable=f"{name}_node", name=f"{name}_node",
            parameters=[params], output="screen",
            condition=IfCondition(LaunchConfiguration(name)),
        )

    bridge_node = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="camera_bridge",
        parameters=[{"config_file": bridge}], output="screen",
        condition=IfCondition(LaunchConfiguration("bridge")),
    )
    return LaunchDescription(args + [bridge_node] + [layer(n) for n in layers])
