cmake_minimum_required(VERSION 3.8)
project(agentic_uav_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox3D.msg"
  "msg/SceneNode.msg"
  "msg/SceneEdge.msg"
  "msg/ProbabilisticSceneGraph.msg"
  "msg/AgentCommand.msg"
  "msg/FlightTelemetry.msg"
  "srv/RAGQuery.srv"
  "srv/RAGInsert.srv"
  "srv/ExecuteMCPTool.srv"
  "srv/ListMCPTools.srv"
  "srv/TriggerReActCycle.srv"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
