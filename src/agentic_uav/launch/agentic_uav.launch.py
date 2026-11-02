#!/usr/bin/env python3
"""
Master Launch File for 5-Layer Agentic UAV System.
Orchestrates:
1. Gazebo Harmonic Simulation & ROS-GZ Bridge
2. Perception Layer (YOLOv11 + Probabilistic 3D Scene Graph)
3. Reasoning Layer (Gemini API Cognitive Engine + ReAct Loop)
4. Action Layer (PX4 SITL Interface & Tool Actuators)
5. Integration Layer (Model Context Protocol MCP Secure Orchestrator)
6. Learning Layer (RAG Cross-Mission Memory)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("agentic_uav")
    params_file = os.path.join(pkg_dir, "config", "agentic_uav_params.yaml")
    world_file = os.path.join(pkg_dir, "worlds", "search_and_rescue_harmonic.sdf")

    # Prefer live mounted files from workspace src if available
    src_world = "/ros2_ws/src/agentic_uav/worlds/search_and_rescue_harmonic.sdf"
    if os.path.isfile(src_world):
        world_file = src_world

    src_params = "/ros2_ws/src/agentic_uav/config/agentic_uav_params.yaml"
    if os.path.isfile(src_params):
        params_file = src_params

    # Launch Configurations
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_gazebo = LaunchConfiguration("launch_gazebo")
    headless = LaunchConfiguration("headless")
    gemini_api_key = LaunchConfiguration("gemini_api_key")

    return LaunchDescription([
        # Launch Arguments
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation clock from Gazebo Harmonic"
        ),
        DeclareLaunchArgument(
            "launch_gazebo",
            default_value="true",
            description="Launch Gazebo Harmonic simulator"
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run Gazebo Harmonic in headless mode (-s)"
        ),
        DeclareLaunchArgument(
            "gemini_api_key",
            default_value="",
            description="Google Gemini API key for Reasoning Layer"
        ),

        # Static World-to-Map Transform Publisher
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_map_tf",
            arguments=["0", "0", "0", "0", "0", "0", "world", "map"],
            parameters=[{"use_sim_time": use_sim_time}]
        ),

        # Gazebo Harmonic Process Execution
        ExecuteProcess(
            condition=IfCondition(launch_gazebo),
            cmd=[
                "gz", "sim",
                world_file,
                "-r"  # Run simulation immediately on startup
            ],
            output="screen"
        ),

        # ROS-GZ Bridge
        # The VelocityControl plugin defaults to /model/agentic_uav/cmd_vel
        # so we bridge that topic. We also bridge /cmd_vel for convenience.
        # The "@" means bidirectional, "[" means GZ->ROS, "]" means ROS->GZ.
        Node(
            condition=IfCondition(launch_gazebo),
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="ros_gz_bridge",
            arguments=[
                # Velocity command: bridge the GZ default topic for this model
                "/model/agentic_uav/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                # Odometry from Gazebo to ROS
                "/uav/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                "/model/agentic_uav/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                # Camera feeds from Gazebo to ROS
                "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                # Simulation clock
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            ],
            remappings=[
                ("/camera/image", "/camera/image_raw"),
                ("/camera/depth_image", "/camera/depth/image_raw"),
            ],
            parameters=[
                {"use_sim_time": use_sim_time}
            ],
            output="screen"
        ),

        # 1. Perception Layer Node
        Node(
            package="agentic_uav",
            executable="perception_node",
            name="perception_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen"
        ),

        # 2. Reasoning Layer Node
        Node(
            package="agentic_uav",
            executable="reasoning_node",
            name="reasoning_node",
            parameters=[
                params_file,
                {
                    "use_sim_time": use_sim_time,
                    "gemini_api_key": gemini_api_key
                }
            ],
            output="screen"
        ),

        # 3. Action Layer Node
        Node(
            package="agentic_uav",
            executable="action_node",
            name="action_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen"
        ),

        # 4. Integration Layer Node (MCP Registry)
        Node(
            package="agentic_uav",
            executable="mcp_integration_node",
            name="mcp_integration_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen"
        ),

        # 5. Learning Layer Node (RAG Memory)
        Node(
            package="agentic_uav",
            executable="learning_node",
            name="learning_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen"
        ),

        # 6. Direct ROS-to-Gazebo Action Relay
        ExecuteProcess(
            cmd=["python3", "/ros2_ws/src/ros_to_gz_relay.py"],
            output="screen"
        )
    ])
