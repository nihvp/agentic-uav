# Agentic UAV

A five-layer agentic UAV stack (perception, reasoning, action, integration, learning)
on ROS 2 Jazzy, PX4 SITL and Gazebo Harmonic, run with Docker Compose on a remote
Ubuntu VM with an NVIDIA GPU, with a local LLM served by Ollama.

> **Status: rebuild in progress.** The v0 prototype is preserved on branch
> `archive/v0`. See [docs/REBUILD_PLAN.md](docs/REBUILD_PLAN.md) for the phases and
> what is implemented so far.

| Layer       | Node             | State                                                       |
|-------------|------------------|-------------------------------------------------------------|
| Action      | `action_node`    | PX4 offboard control, `NavigateTo` action, `FlightCommand` service (Phase 1, flown in SITL) |
| Reasoning   | `reasoning_node` | Provider-agnostic LLM tool loop; local LLM via OpenAI-compatible adapter, Gemini optional (Phase 3) |
| Perception  | `perception_node`| Colour-blob detector + depth + PX4 pose -> scene graph (Phase 2) |
| Learning    | `learning_node`  | Persistent RAG store seeded with SOPs (Phase 5)             |
| Integration | `mcp_server`     | MCP server with **mock** tools, not yet wired in (Phase 4)  |

## Layout

```
compose.yaml, compose.gpu.yaml   sim, agent and llm services; NVIDIA GPU overlay
docker/                          sim (PX4 + Gazebo + XRCE agent) and agent (ROS 2) images
scripts/check_host.sh            checks GPU, Docker, NVIDIA toolkit, display, ports
scripts/run.sh                   launch (adds the GPU overlay when available)
src/agentic_uav_interfaces/      msgs, srvs, actions
src/agentic_uav/                 Python nodes, one sub-package per layer, unit tests in test/
src/agentic_uav_bringup/         launch, params, Gazebo bridge config, worlds
docs/REBUILD_PLAN.md             architecture and phased plan
```

## Run on the VM

Setup, daily workflow and troubleshooting: [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md).

```bash
cp .env.example .env              # LLM_MODEL etc.
scripts/check_host.sh --gpu-test
scripts/run.sh --build            # first run builds the images and pulls the model
```

Options: `--headless` (no Gazebo GUI), `--software-gl`, `--sim-only`, `--no-llm`.

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
docker compose exec agent bash -lc 'source /ws/install/setup.bash && ros2 service call /reasoning/run_mission agentic_uav_interfaces/srv/RunMission "{goal: \"Find the survivor and hover above them.\"}"'
```

## Tests

The unit tests are plain Python with no ROS or simulator needed:

```bash
cd src/agentic_uav && python3 -m pytest test
```
