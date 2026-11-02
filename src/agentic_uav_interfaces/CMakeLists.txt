cmake_minimum_required(VERSION 3.8)
project(agentic_uav_interfaces)

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
  "msg/FlightTelemetry.msg"
  "srv/FlightCommand.srv"
  "srv/RAGQuery.srv"
  "srv/RAGInsert.srv"
  "srv/RunMission.srv"
  "action/NavigateTo.action"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
