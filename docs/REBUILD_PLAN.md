# Agentic UAV — Rebuild Plan

The v0 prototype is preserved on branch `archive/v0`. This branch (`rebuild`) starts
from a clean skeleton. The guiding rule for every phase: **no layer may read simulator
ground truth or fake an outcome.** If something is not implemented yet, it fails loudly
or reports "not available" to the reasoning layer; it never returns canned success.

## Target stack

| Concern        | Choice                                                                 |
|----------------|------------------------------------------------------------------------|
| OS (container) | Ubuntu 24.04 (Noble)                                                   |
| ROS 2          | Jazzy (supported until May 2029)                                       |
| Simulator      | Gazebo Harmonic (the official pairing for Jazzy)                       |
| Autopilot      | PX4 SITL, `gz_x500_depth` airframe (quadrotor + RGB-D camera)          |
| PX4 <-> ROS 2  | uXRCE-DDS (MicroXRCEAgent) + `px4_msgs`, versions pinned to the PX4 tag |
| LLM            | Provider-agnostic adapter; Gemini first, others pluggable              |
| Tools          | Real MCP server (`mcp` Python SDK) for external tools                  |
| Memory         | Persistent vector store on a mounted volume                            |
| Runtime        | Docker Compose on the Fedora VM (aarch64), host networking             |

## Architecture

```
            mission goal (service / CLI)
                       |
                       v
  +------------------ reasoning_node ------------------+
  |  AgentLoop: LLM provider + tool registry           |
  |  tool call -> ROS action/service -> real result    |
  +---+--------------+---------------+--------------+--+
      |              |               |              |
 scene graph    rag_query       MCP client     NavigateTo action /
 (topic)        (service)       (stdio)        FlightCommand service
      |              |               |              |
 perception_node learning_node  mcp_server     action_node  <-- sole owner of PX4 control
      ^                                             |
      | camera / depth (ros_gz_bridge)              | px4_msgs over uXRCE-DDS
      +------------ Gazebo Harmonic + PX4 SITL -----+
```

Key differences from v0:

* **One control owner.** Only `action_node` sends PX4 setpoints. No relay, no direct
  `gz` velocity commands, no CLI subprocesses.
* **Tools return real results.** The LLM calls a tool; the tool is a ROS action or
  service whose response (success, failure, feedback) becomes the observation.
* **Perception sees only sensor data.** Camera + depth + PX4 odometry. No `gz model`
  polling, no name-based labelling.
* **No hard-coded world coordinates** outside the world SDF.

## Package layout

```
compose.yaml                     sim + agent services
docker/sim.Dockerfile            PX4 SITL + Gazebo Harmonic + MicroXRCEAgent
docker/agent.Dockerfile          ROS 2 Jazzy + px4_msgs + this workspace
scripts/run_on_fedora.sh         host prep (X11, GPU check) + compose up
src/agentic_uav_interfaces/      msgs, srvs, actions (NavigateTo, FlightCommand, ...)
src/agentic_uav/                 Python package, one sub-package per layer
  common/frames.py               ENU <-> NED conversions (PX4 is NED)
  action/                        offboard state machine + action_node
  perception/                    detector, scene graph builder, perception_node
  reasoning/                     llm/ adapters, tools, agent loop, reasoning_node
  integration/                   MCP server + client bridge
  learning/                      vector store, SOP documents, learning_node
  test/                          pure-Python unit tests (no ROS needed)
src/agentic_uav_bringup/         launch files, params, bridge config, worlds
```

## Phases

Each phase ends with a demo you can run and an acceptance check. Do not start a phase
until the previous one passes.

### Phase 0 — Environment (skeleton is in place; needs verification on the VM)

* Build both images on the Fedora VM (`scripts/run_on_fedora.sh --build`).
* `sim` service: PX4 SITL starts with the `search_and_rescue` world, Gazebo GUI or
  headless.
* `agent` service: `ros2 topic list` shows `/fmu/out/*` topics from PX4.

**Accept:** `ros2 topic echo /fmu/out/vehicle_local_position --once` prints data inside
the agent container. Note the exact `vehicle_status` topic name (versioned in recent
PX4 releases, e.g. `vehicle_status_v1`) and set it in `params.yaml`.

**Risk:** UTM on Apple Silicon has weak GPU passthrough. If the RGB-D sensor stalls,
run Gazebo headless with `LIBGL_ALWAYS_SOFTWARE=1`, lower camera resolution / rate in
the world, and use the GUI only for spot checks.

### Phase 1 — Action layer (vertical slice)

* `action_node` (skeleton implemented): streams `OffboardControlMode` +
  `TrajectorySetpoint` at 10–20 Hz, arms, switches to offboard, takes off.
* Expose `NavigateTo` action (goal ENU position, feedback distance, result reached /
  aborted) and `FlightCommand` service (TAKEOFF, HOLD, LAND, RTL).
* Geofence enforced in `action_node`, plus PX4's own geofence params.
* Publish `FlightTelemetry` from PX4 status + local position.

**Accept:** `ros2 action send_goal /action/navigate_to ...` flies the x500 to the
point and returns success. RTL and LAND work. Killing the agent container triggers
PX4 offboard-loss failsafe (not a crash).

### Phase 2 — Perception layer

* Bridge the x500_depth camera topics (confirm names with `gz topic -l`).
* Detector: YOLO (Ultralytics) for `person`; fire / debris / landing pad need either a
  fine-tuned model or simple, honest colour/shape detectors clearly labelled as such.
* Back-projection uses camera intrinsics + camera extrinsic from the model SDF + PX4
  attitude (convert NED -> ENU via `common/frames.py`).
* Reuse `scene_graph_builder.py` (log-odds existence, covariance, relation edges).

**Accept:** Hovering 5 m short of the survivor, facing it (the camera is pitched 45°
down, so it cannot see straight below), yields one confirmed `survivor` node within
~1 m of its true SDF position, and the node decays when out of view.

**Implemented (memory-constrained VM):** `ColorDetector` (HSV blobs matched to the
world's solid colours, every node tagged `detector:color`) instead of YOLO, which
would need torch in RAM. The sim patches the x500_depth camera to 45° pitch and
640x360 @ 10 Hz RGB. Debris (grey on grey) is not detected. YOLO plus a Fuel person
model remains an option once the VM has more memory.

**Measured in SITL (hover at (3, 4, 5)):** survivor placed 0.3-0.5 m from its SDF
position (passes). Known error sources, checked against Gazebo ground truth for
diagnosis only: PX4's estimated yaw was 3.6° off the true heading and its altitude
0.3 m high, which rotates/shifts every projected point; depth sees an object's near
surface, not its centre; objects cut by the image edge are placed at the centroid of
their visible part (the landing pad seen at 50° off-axis landed 1.7 m off).

### Phase 3 — Reasoning layer

* `LLMProvider` interface (`complete(messages, tools) -> text | tool_calls`) with a
  Gemini adapter using native function calling. A `ScriptedProvider` exists only for
  tests and must be selected explicitly.
* Tool registry is the single source of truth for tool schemas; every tool maps to a
  ROS action/service call and returns its real result.
* `AgentLoop`: multi-step ReAct with message history, max-steps budget, per-tool
  timeouts, and a safety gate (e.g. no navigation outside geofence, battery checks).
* Missions are started with the `RunMission` service (goal text) and
  report a trace (thoughts, tool calls, results).

**Accept:** "Find the survivor and hover above them" completes using only perception
output and the navigation tool, with the full trace logged.

### Phase 4 — Integration layer (MCP)

* `integration/mcp_server.py`: a real MCP server (stdio) exposing external tools
  (weather, airspace clearance, alert dispatch). Mocked back-ends are fine, but they
  must be labelled as mocks in their tool descriptions.
* Reasoning connects as an MCP client and merges MCP tools into its registry.

**Accept:** `mcp` inspector lists the tools; the agent calls one during a mission.

### Phase 5 — Learning layer

* Vector store with `sentence-transformers` embeddings, persisted to `/data` (volume).
  The fallback embedder must be deterministic (no Python `hash()`).
* `rag_query` / `rag_insert` services; seed with `sop_documents.py`.
* After each mission, the agent writes an after-action summary back to the store.

**Accept:** Restarting containers keeps inserted documents; a second mission retrieves
the first mission's report.

### Phase 6 — Evaluation

* Scenario runner: spawn survivors / hazards at random positions (via world
  generation, not mid-mission teleporting), run N missions, record success rate, time,
  tool calls, and LLM tokens.
* CI: unit tests in `src/agentic_uav/test` run without ROS or simulator.

## What was salvaged from v0

| v0 file                                          | Now                                                |
|--------------------------------------------------|----------------------------------------------------|
| `perception/scene_graph_builder.py`              | Kept; ground-truth injection (`update_world_entity`) removed |
| `learning/default_sop_knowledge.py`              | `learning/sop_documents.py`                        |
| `agentic_uav_interfaces` msgs/srvs               | Kept; `AgentCommand` removed in favour of actions  |
| `worlds/search_and_rescue_harmonic.sdf`          | Objects kept; fake drone removed; PX4 systems added |
| ReAct system prompt                              | Basis for `reasoning/prompts.py`                   |

Deleted outright: `ros_to_gz_relay.py` (competing controller + survivor teleporting),
`GazeboWorldMonitor` (ground-truth perception), the regex "offline planner",
`mcp_server_registry.py` (canned responses), and the ad-hoc shell scripts.

## Open items to verify on the VM

1. PX4 tag that builds cleanly on aarch64 Ubuntu 24.04 (default pin: `v1.16.0`), and
   the matching `px4_msgs` branch.
2. Versioned PX4 topic names (`vehicle_status` vs `vehicle_status_v1`, etc.).
3. Gazebo camera topic names for `x500_depth`.
4. Whether `PX4_GZ_WORLD` picks up the mounted custom world (it must be in PX4's
   `Tools/simulation/gz/worlds`, which the sim entrypoint symlinks).
