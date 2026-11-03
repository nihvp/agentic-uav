# Agentic UAV

A five-layer agentic UAV stack (perception, reasoning, action, integration, learning)
on ROS 2 Jazzy, PX4 SITL and Gazebo Harmonic, run with Docker Compose on a Fedora VM.

> **Status: rebuild in progress.** The v0 prototype is preserved on branch
> `archive/v0`. See [docs/REBUILD_PLAN.md](docs/REBUILD_PLAN.md) for the phases and
> what is implemented so far.

| Layer       | Node             | State                                                       |
|-------------|------------------|-------------------------------------------------------------|
| Action      | `action_node`    | PX4 offboard control, `NavigateTo` action, `FlightCommand` service (Phase 1, flown in SITL) |
| Reasoning   | `reasoning_node` | Provider-agnostic LLM tool loop; Gemini adapter (Phase 3)   |
| Perception  | `perception_node`| Camera pipeline + scene graph; no detector yet (Phase 2)    |
| Learning    | `learning_node`  | Persistent RAG store seeded with SOPs (Phase 5)             |
| Integration | `mcp_server`     | MCP server with **mock** tools, not yet wired in (Phase 4)  |

## Layout

```
compose.yaml, compose.gpu.yaml   sim + agent services
docker/                          sim (PX4 + Gazebo + XRCE agent) and agent (ROS 2) images
scripts/run_on_fedora.sh         host prep and launch
src/agentic_uav_interfaces/      msgs, srvs, actions
src/agentic_uav/                 Python nodes, one sub-package per layer, unit tests in test/
src/agentic_uav_bringup/         launch, params, Gazebo bridge config, worlds
docs/REBUILD_PLAN.md             architecture and phased plan
```

## Run on the Fedora VM

```bash
export GEMINI_API_KEY=...        # needed by reasoning_node
scripts/run_on_fedora.sh --build # first run builds both images (PX4 build takes a while)
```

Options: `--headless` (no Gazebo GUI), `--software-gl` (force llvmpipe if the VM GPU
struggles), `--sim-only` (just PX4 + Gazebo).

### Phase 1 smoke test

In another terminal:

```bash
docker compose exec agent bash -lc 'source /ws/install/setup.bash && ros2 topic list | grep fmu'
```

```bash
docker compose exec agent bash -lc 'source /ws/install/setup.bash && ros2 service call /action/flight_command agentic_uav_interfaces/srv/FlightCommand "{command: TAKEOFF, altitude: 5.0}"'
```

```bash
docker compose exec agent bash -lc 'source /ws/install/setup.bash && ros2 action send_goal --feedback /action/navigate_to agentic_uav_interfaces/action/NavigateTo "{target: {x: 8.0, y: 4.0, z: 5.0}, yaw: .nan, acceptance_radius: 0.0}"'
```

### Run an LLM mission

```bash
docker compose exec agent bash -lc 'source /ws/install/setup.bash && ros2 service call /reasoning/run_mission agentic_uav_interfaces/srv/RunMission "{goal: \"Take off to 5 m, fly to (8, 4, 5), hover, then return to launch.\"}"'
```

## Tests

The unit tests are plain Python with no ROS or simulator needed:

```bash
cd src/agentic_uav && python3 -m pytest test
```
