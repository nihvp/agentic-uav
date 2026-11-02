cmake_minimum_required(VERSION 3.8)
project(agentic_uav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(agentic_uav_interfaces REQUIRED)

find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install node executable scripts
install(PROGRAMS
  scripts/perception_node
  scripts/reasoning_node
  scripts/action_node
  scripts/mcp_integration_node
  scripts/learning_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch, configuration, and simulation worlds
install(DIRECTORY
  launch
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(agentic_uav_interfaces)
ament_package()
