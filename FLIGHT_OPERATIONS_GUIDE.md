# 5-Layer Agentic UAV: Complete Setup & Flight Operations Manual

A comprehensive, production-ready operational guide for the **5-Layer Agentic UAV Architecture** operating on **ROS 2 Humble** and **Gazebo Harmonic**.

---

## System Architecture

```text
                      ┌────────────────────────────────────────┐
                      │      Reasoning Layer (Gemini LLM)      │
                      │  Thought ──> Action ──> Observation    │
                      └────▲───────────────────▲───────────┬───┘
                           │                   │           │
          ┌────────────────┴──────────┐        │           │
          │  Learning Layer (RAG)     │        │           │
          │  SOP Vector Memory Store  │        │           │
          └───────────────────────────┘        │           │
                           ┌───────────────────┴───────┐   │
                           │ Perception Layer (YOLOv11)│   │
                           │ 3D Scene Graph Builder    │   │
                           └───────────────────▲───────┘   │
                                               │           ▼
┌──────────────────────────────────────────────┴───────────────┐
│              Action Layer (PX4 SITL Controller)              │
│       Flight State Machine & Tool Actuator Dispatch          │
└──────────────────────────────┬───────────────────────────────┘
                               │ /model/agentic_uav/cmd_vel
                               ▼
┌──────────────────────────────────────────────────────────────┐
│            Gazebo Harmonic Simulation Environment            │
│  Physics + RGB-D Camera + Ground Plane + Search & Rescue     │
└──────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Boot & Environment Startup

### 1. Boot Your Fedora VM & Open Terminal 1
Navigate to the repository in your shared folder:

```bash
cd /mnt/utm-share/agentic-uav
```
*(Adjust the path if your UTM shared directory is mapped elsewhere).*

### 2. (Optional) Set Google Gemini API Key
If you want to use live multimodal cloud AI reasoning:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
> **Note**: If left unset, the system automatically uses its embedded **Deterministic ReAct Engine** with zero internet dependency or API costs.

### 3. Launch Simulation & 5-Layer Stack
Run the clean deployment script:
```bash
./run_on_fedora.sh
```

**What happens automatically:**
* Configures local display permissions (`xhost +local:root`) for GPU hardware acceleration.
* Completely cleans up any stale background Docker containers (`--remove-orphans`).
* Builds and boots a fresh container running ROS 2 Humble and Gazebo Harmonic.
* Opens the **Gazebo Sim 3D simulation window** with the quadrotor resting on the ground plane.

---

## Phase 2: Opening the Operator Terminal

Open a **second terminal window** (Terminal 2) in your Fedora VM:

```bash
docker exec -it agentic_uav_container bash
```

Inside this operator terminal, start the high-speed Action Relay:
```bash
python3 /ros2_ws/src/ros_to_gz_relay.py &
```

Verify that all nodes are online:
```bash
ros2 node list
```

Expected output:
```text
/action_node
/learning_node
/mcp_integration_node
/perception_node
/reasoning_node
/ros_gz_bridge
/ros_to_gz_relay
/world_to_map_tf
```

---

## Phase 3: Flight Control Methods

### Method A: Autonomous Cognitive ReAct AI Mission

Trigger the complete cognitive reasoning loop:

```bash
ros2 service call /reasoning/trigger_react_cycle agentic_uav_interfaces/srv/TriggerReActCycle "{
  mission_goal: 'Conduct search and rescue reconnaissance over Sector Alpha: locate survivors, assess thermal hazards, and verify emergency landing clearance.'
}"
```

**What the AI does automatically:**
1. **Perception**: Scans the 3D semantic scene graph, locating the survivor entity and wildfire hazard.
2. **Reasoning (ReAct)**: Evaluates threats, dispatches an emergency alert tool via the Model Context Protocol (MCP) layer, and calculates the optimal aerial reconnaissance coordinates.
3. **Flight**: The UAV lifts off, cruises across the terrain to coordinates $(8.0, 4.0, 4.0)$, and establishes visual coverage above the survivor.
4. **Evacuation**: After 2.5 seconds of confirmed aerial coverage, the ground rescue squad moves the survivor out of the wildfire zone to the **Emergency Landing Pad (the blue circle at $0.0, 8.0$)** in plain view.

---

### Method B: Autonomous 3D Waypoint Navigation

You can command the autopilot to fly directly to any $(X, Y, Z)$ coordinates at a specified speed:

#### 1. Fly to Survivor Reconnaissance Position
```bash
ros2 topic pub --once /action/agent_command agentic_uav_interfaces/msg/AgentCommand "{
  action_type: 'NAVIGATE_WAYPOINT',
  target_position: {x: 8.0, y: 4.0, z: 4.0},
  speed: 2.0
}"
```

#### 2. Fly to the Emergency Landing Pad (Blue Circle)
```bash
ros2 topic pub --once /action/agent_command agentic_uav_interfaces/msg/AgentCommand "{
  action_type: 'NAVIGATE_WAYPOINT',
  target_position: {x: 0.0, y: 8.0, z: 2.5},
  speed: 2.0
}"
```

#### 3. Touch Down / Land on the Pad
```bash
ros2 topic pub --once /action/agent_command agentic_uav_interfaces/msg/AgentCommand "{
  action_type: 'NAVIGATE_WAYPOINT',
  target_position: {x: 0.0, y: 8.0, z: 0.3},
  speed: 1.0
}"
```

#### 4. Return to Launch Base (RTL)
```bash
ros2 topic pub --once /action/agent_command agentic_uav_interfaces/msg/AgentCommand "{
  action_type: 'RTL'
}"
```

---

### Method C: Manual Flight Control

#### 1. Interactive Keyboard Flight (Teleop)
Steer the UAV like a game controller:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

* **`i`**: Fly forward
* **`,`**: Reverse backward
* **`j` / `l`**: Steer left / right
* **`t` / `b`**: Climb / descend
* **`k` or `Space`**: Stop and hold hover

#### 2. Continuous Velocity Streaming (`/cmd_vel`)
Stream continuous velocity from the command line:
```bash
# Fly forward with vertical lift:
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.5, y: 0.0, z: 0.5}}"
```
*(Press `Ctrl+C` to stop streaming).*

To immediately halt momentum and hold hover:
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

## Phase 4: Operational Utilities & Helper Scripts

All helper scripts are located in `/ros2_ws/src/`:

| Script | Purpose |
| :--- | :--- |
| `bash /ros2_ws/src/launch_and_test.sh` | **1-Click Test**: Starts the relay and flies the drone to waypoint $(8.0, 4.0, 4.0)$ |
| `bash /ros2_ws/src/reset_and_fly.sh` | Teleports the drone directly in front of the camera at $(0, 0, 2.0)$ and tests motion |
| `bash /ros2_ws/src/reset_survivor.sh` | Resets the survivor model back to the fire hazard zone for repeatable missions |
| `bash /ros2_ws/src/reload_ai.sh` | Hot-reloads the Reasoning Layer and Relay in the container |
| `bash /ros2_ws/src/diagnose.sh` | Runs an automated health check on all ROS nodes, topics, and bridges |

---

## Autonomous Background Loop Control

By default, the Reasoning Layer continuously re-evaluates the 3D scene graph. If you want the AI to **only** deliberate when you explicitly invoke it (preventing background commands from competing with manual flight):

```bash
ros2 param set /reasoning_node auto_cycle_enabled false
```
