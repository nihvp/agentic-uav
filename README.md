# 5-Layer Agentic UAV System (ROS 2 Humble + Gazebo Harmonic)

An enterprise-grade, cognitive autonomous drone architecture built on **ROS 2 Humble**, **Gazebo Harmonic**, and **PX4 SITL**, integrating multimodal vision, cognitive LLM reasoning (**Google Gemini API**), atomic tool orchestration via the **Model Context Protocol (MCP)**, and cross-mission **Retrieval-Augmented Generation (RAG)** memory.

Containerized with Docker on Ubuntu 22.04 LTS and pre-configured for hardware-accelerated GUI pass-through on **Fedora VM hosts**.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |            MISSION GOAL               |
                      |  (Search & Rescue / Hazard Survey)    |
                      +-------------------+-------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------+
|                                 REASONING LAYER                                 |
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   |                        Gemini 2.5 Cognitive Engine                      |   |
|   |                         ReAct Loop Orchestrator                         |   |
|   |  Thought  -->  Action Selection  -->  Action Input  -->  Observation    |   |
|   +-------------------------------------------------------------------------+   |
+---------^-----------------------^---------------------^------------------+------+
          |                       |                     |                  |
          | Probabilistic         | SOP Retrieval       | MCP Tool         | Flight & Tool
          | 3D Scene Graph        | & Episodic Memory   | Schemas & Calls  | Commands
          |                       |                     |                  |
+---------+----------+  +---------+----------+  +-------+----------+  +----v------+---+
|  PERCEPTION LAYER  |  |   LEARNING LAYER   |  | INTEGRATION L.   |  | ACTION LAYER  |
|                    |  |                    |  | (MCP Engine)     |  |               |
| * Simulated RGB-D  |  | * Semantic Vector  |  | * MCP Registry   |  | * PX4 SITL    |
|   Sensor Feeds     |  |   Store & Cosine   |  | * Schema Valid.  |  |   Offboard    |
| * YOLOv11 Detector |  |   Embeddings       |  | * Token Auth     |  |   Controller  |
| * 3D Back-Project. |  | * Standard Ops     |  | * External APIs: |  | * Geofencing  |
| * Bayesian State   |  |   Procedures (SOP) |  |   - CAD Dispatch |  | * Tool        |
| * Spatial Graph &  |  | * Cross-Mission    |  |   - Airspace UTM |  |   Actuators   |
|   Relational Edges |  |   Episodic Memory  |  |   - Weather Feed |  |   (Gimbal,    |
+---------^----------+  +--------------------+  +------------------+  |    Payload)   |
          |                                                           +-------+-------+
          | GZ Topics                                                         | Setpoints
          | (Camera, Depth, Odom)                                             | (Twist/Pose)
+---------+-------------------------------------------------------------------v-------+
|                                SIMULATION ENVIRONMENT                               |
|                  Gazebo Harmonic (GZ Sim) + PX4 Autopilot SITL                      |
|              World: search_and_rescue_harmonic.sdf (Victims, Hazards, LZ)           |
+-------------------------------------------------------------------------------------+
```

---

## 5-Layer Specifications

### 1. Perception Layer (`perception_node`)
- **Sensors Ingested**: Simulated RGB (`/camera/image_raw`), Depth (`/camera/depth/image_raw`), Intrinsics (`/camera/camera_info`), and Drone Odometry (`/uav/odometry`).
- **Object Detection**: YOLOv11 (`yolo11n.pt` with automatic resilient synthetic fallback for offline simulation environments).
- **3D Spatial Back-Projection**: Inverse pinhole camera model $(u, v, d) \to (X_c, Y_c, Z_c)$ transformed to world frame via camera-to-body extrinsics and drone orientation.
- **Bayesian Track & Covariance**: Recursive log-odds existence probability filtering ($P(\text{exists})$) and depth-dependent 3D spatial covariance.
- **Relational Scene Graph**: Evaluates topological and metric spatial relationships (`near`, `threatens`, `blocking_path`, `inside_zone`) with associated uncertainty probabilities.
- **Publishes**: `/perception/scene_graph` (`agentic_uav_interfaces/msg/ProbabilisticSceneGraph`) and `/perception/scene_graph_markers` (`visualization_msgs/msg/MarkerArray`).

### 2. Reasoning Layer (`reasoning_node`)
- **Cognitive Engine**: Google Gemini API (`gemini-2.5-flash`) with structured prompt framing and offline deterministic ReAct fallback.
- **ReAct Workflow**:
  - `Thought`: Evaluates scene nodes, threats, relations, and vehicle telemetry.
  - `Action`: Chooses from allowed atomic actions (`navigate_waypoint`, `inspect_target`, `query_rag_memory`, `invoke_mcp_tool`, `deploy_payload`, `return_to_launch`, `mission_complete`).
  - `Action Input`: Strictly validated JSON parameters.
  - `Observation`: Dynamic feedback loop from RAG queries, MCP tools, and flight status.
- **Publishes**: `/action/agent_command` (`agentic_uav_interfaces/msg/AgentCommand`).
- **Services**: `/reasoning/trigger_react_cycle` (`agentic_uav_interfaces/srv/TriggerReActCycle`).

### 3. Action Layer (`action_node`)
- **PX4 SITL Bridge**: Streams offboard trajectory setpoints (`/fmu/in/trajectory_setpoint`, `/mavros/setpoint_position/local`, `/cmd_vel`) at 20 Hz.
- **Flight State Machine**: `DISARMED` $\to$ `ARMING` $\to$ `TAKEOFF` $\to$ `OFFBOARD` $\to$ `LOITER` $\to$ `RTL` $\to$ `LANDED`.
- **Safety Geofence**: Strict ceiling (25m) and horizontal radius (150m) enforcement with automatic Return-to-Launch failsafe.
- **Tool Actuators**: Executes physical/digital tool requests (camera gimbal pitch/pan, thermal sensor activation, emergency strobe, payload release solenoid).
- **Publishes**: `/action/flight_telemetry` (`agentic_uav_interfaces/msg/FlightTelemetry`).

### 4. Integration Layer (`mcp_integration_node`)
- **Model Context Protocol (MCP)**: Implements standardized tool discovery (`ListMCPTools`) and execution (`ExecuteMCPTool`) adhering to JSON Schema guidelines.
- **Security & Authorization**: Bearer token validation and parameter verification before execution.
- **Integrated External Connectors**:
  - `emergency_alert_dispatch`: Dispatches incident alerts to Computer-Aided Dispatch (CAD) / Emergency Operations Center.
  - `utm_airspace_clearance`: Dynamic UTM flight corridor reservations and altitude clearance.
  - `weather_advisory_feed`: Local meteorological updates (wind gusts, visibility, VFR conditions).
  - `satellite_imagery_query`: Pre-disaster reference imagery comparison.

### 5. Learning Layer (`learning_node`)
- **Retrieval-Augmented Generation (RAG)**: Ingests Standard Operating Procedures (SOPs), safety regulations, and cross-mission lessons learned.
- **Semantic Vector Store**: Vector embedding engine with cosine similarity search and category filtering (`SOP`, `REGULATION`, `CHECKLIST`, `MISSION_LOG`).
- **Services**:
  - `/learning/rag_query`: Queries knowledge base for procedures matching environmental state.
  - `/learning/rag_insert`: Ingests live mission findings, obstacle positions, and after-action reports for cross-mission retention.

---

## Directory Layout

```
.
├── Dockerfile                  # Container definition (ros:humble-desktop + Gazebo Harmonic)
├── docker-compose.yml          # Container configuration with host networking & DRI acceleration
├── entrypoint.sh               # Runtime initialization script
├── run_on_fedora.sh            # Automated deployment helper script for Fedora VM hosts
├── requirements.txt            # Python dependencies (YOLOv11, Gemini SDK, MCP, Transformers)
├── tests/
│   └── test_agentic_uav.py     # End-to-end unit and integration test suite
└── src/
    ├── agentic_uav_interfaces/ # ROS 2 Interfaces Package (Messages & Services)
    │   ├── CMakeLists.txt      # Message & service generation build rules
    │   ├── package.xml         # Interface package manifest
    │   ├── msg/                # Custom ROS 2 message definitions
    │   │   ├── BoundingBox3D.msg
    │   │   ├── SceneNode.msg
    │   │   ├── SceneEdge.msg
    │   │   ├── ProbabilisticSceneGraph.msg
    │   │   ├── AgentCommand.msg
    │   │   └── FlightTelemetry.msg
    │   └── srv/                # Custom ROS 2 service definitions
    │       ├── RAGQuery.srv
    │       ├── RAGInsert.srv
    │       ├── ExecuteMCPTool.srv
    │       ├── ListMCPTools.srv
    │       └── TriggerReActCycle.srv
    └── agentic_uav/            # Core ROS 2 Application Package
        ├── CMakeLists.txt      # ROS 2 CMake build file with Python install & scripts
        ├── package.xml         # Package manifest and dependencies
        ├── setup.py            # Python setuptools packaging
        ├── agentic_uav/        # Python library implementation
        │   ├── perception/     # YOLOv11 & 3D Scene Graph Builder
        │   ├── reasoning/      # Gemini API & ReAct Workflow Engine
        │   ├── action/         # PX4 Controller & Tool Actuators
        │   ├── integration/    # MCP Tool Registry & Security Engine
        │   ├── learning/       # Vector Store & RAG SOP Knowledge
        │   └── utils/          # Transform and spatial math utilities
        ├── scripts/            # Node entrypoint executables
        ├── launch/             # ROS 2 launch files integrating Gazebo Harmonic
        ├── config/             # YAML configurations for nodes and ros_gz_bridge
        └── worlds/             # Gazebo Harmonic SDF world definition
```

---

## Deployment on Fedora VM Host

### Prerequisites on Fedora Host
1. Install Docker and Docker Compose:
   ```bash
   sudo dnf install -y docker docker-compose
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
2. Enable local display pass-through (X11 / Wayland):
   ```bash
   xhost +local:root
   ```

### Quick Start with Docker Compose
1. Clone the repository and export your Gemini API key (optional for live AI reasoning):
   ```bash
   git clone https://github.com/example/agentic-ai-uav.git
   cd agentic-ai-uav
   export GEMINI_API_KEY="your-gemini-api-key"
   ```
2. Launch the entire system:
   ```bash
   ./run_on_fedora.sh
   # Or directly:
   docker compose up --build
   ```

---

## Native Build & Execution (Inside Container or Ubuntu 22.04 Host)

```bash
# 1. Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# 2. Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# 3. Build workspace
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# 4. Source local workspace
source install/setup.bash

# 5. Launch all 5 layers with Gazebo Harmonic
ros2 launch agentic_uav agentic_uav.launch.py
```

### Running Headless (Simulation without GUI)
```bash
ros2 launch agentic_uav agentic_uav.launch.py headless:=true
```

---

## Testing & Verification

Run the self-contained test suite covering all 5 layers:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
# Or with pytest:
pytest tests/test_agentic_uav.py -v
```

---

## Interacting with the Live System

### Inspect the 3D Scene Graph
```bash
ros2 topic echo /perception/scene_graph
```

### Inspect Vehicle Flight Telemetry
```bash
ros2 topic echo /action/flight_telemetry
```

### Query RAG Memory Service
```bash
ros2 service call /learning/rag_query agentic_uav_interfaces/srv/RAGQuery "{query_text: 'What is the procedure for an obstructed landing zone?', top_k: 2}"
```

### Discover MCP Registered Tools
```bash
ros2 service call /integration/list_mcp_tools agentic_uav_interfaces/srv/ListMCPTools "{}"
```

### Trigger an On-Demand Cognitive ReAct Cycle
```bash
ros2 service call /reasoning/trigger_react_cycle agentic_uav_interfaces/srv/TriggerReActCycle "{mission_goal: 'Confirm survivor location and verify emergency landing zone clearance.'}"
```

---
