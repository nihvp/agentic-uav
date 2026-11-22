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
| LLM            | Provider-agnostic adapter; local open model on the VM GPU via Ollama's OpenAI-compatible API, Gemini kept as an option |
| Tools          | Real MCP server (`mcp` Python SDK) for external tools                  |
| Memory         | Persistent vector store on a mounted volume                            |
| Runtime        | Docker Compose on a remote Ubuntu VM (x86_64, NVIDIA GPU), host networking; was a Fedora VM (aarch64) up to Phase 3 WIP |

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

## Move to the remote GPU VM (2026-11)

Phases 0-2 and the Phase 3 work so far ran on a Fedora VM in UTM (aarch64, ~4.8 GB
RAM, no usable GPU). Two limits stopped Phase 3 acceptance:

* **LLM quota.** The Gemini free tier allows ~5 requests/min and ~20/day per model, so
  1-2 missions a day with 12 s pacing between steps. Not enough to iterate on prompts.
* **VM capacity.** VM stalls tripped the PX4 offboard failsafe, the camera was cut to
  640x360 and perception ran at 2-3 Hz, and YOLO was dropped (no RAM for torch).

The new host is a remote Ubuntu VM reached by AnyDesk: x86_64, 8 CPUs, 53 GiB RAM,
684 GB free disk, RTX 5070 Ti (16 GB, Blackwell), driver 590 / CUDA 13.1, X11 desktop. Workflow: edit on the Mac, push to GitHub, `git pull` on the VM, run and test
there, bring findings back.

### What changes

| Area | Fedora VM (before) | Remote GPU VM (after) |
|------|--------------------|-----------------------|
| Host script | `scripts/run_on_fedora.sh`, `/dev/dri` passthrough | `scripts/run.sh` + `scripts/check_host.sh` (arch, `nvidia-smi`, NVIDIA Container Toolkit, Docker, disk) |
| GPU in containers | `compose.gpu.yaml` mapped `/dev/dri` | `compose.gpu.yaml` reserves the NVIDIA GPU (`deploy.resources.reservations.devices`) for `sim`, `agent` and `llm` |
| Gazebo rendering | headless or llvmpipe | GPU rendering; GUI on the VM desktop (AnyDesk) or headless |
| LLM | Gemini free tier over the internet | new `llm` service: Ollama serving an open model with tool calling (default `qwen3:14b`, 4-bit, 16k context), OpenAI-compatible API on `127.0.0.1:11434/v1` |
| LLM adapter | `GeminiProvider` only | new `OpenAICompatProvider` (`openai` SDK; also works with Ollama / LM Studio). Gemini stays selectable; pacing defaults to 0 |
| Agent image | CPU-only torch | CUDA 12.8 torch (required for Blackwell), so YOLO and `sentence-transformers` run on the GPU |
| Perception | colour detector only | colour detector stays the default; YOLO for `person` becomes a real option (Phase 2 follow-up) |
| Sim tuning | `COM_OF_LOSS_T=3`, 640x360 @ 10 Hz camera | keep at first, measure, then restore PX4 defaults and a larger camera if the VM keeps up |
| Run guide | Fedora-specific (`/mnt/utm-share`, `sed -i` caveat) | rewritten for the Ubuntu VM and the git-pull workflow |

The layer code and ROS interfaces do not change for the move. Only the LLM adapter,
container setup and docs do. The rule above still holds: no ground truth, no faked
results.

### Order of work

1. **Host bring-up.** `check_host.sh`, `run.sh`, NVIDIA `compose.gpu.yaml`, run guide.
   Build both images on the VM. **Accept:** Phase 1 and Phase 2 checks pass again on
   the new VM (takeoff, `NavigateTo`, RTL; survivor placed within ~1 m).
2. **Local LLM.** `llm` compose service (Ollama, model and context length from `.env`),
   `OpenAICompatProvider` with unit tests against a fake server, `params.yaml` switched
   to `llm.provider: ollama`. Pick the model by GPU memory; it must support
   native tool calling. **Accept:** a scripted single-tool request round-trips through
   Ollama from inside the `agent` container (`python3 -m agentic_uav.reasoning.llm_check`).
3. **Finish Phase 3.** Run "Find the survivor and hover above them" repeatedly (now
   free), tune prompts and tools, record success over ~10 runs. Re-measure whether the
   VM-stall workarounds (offboard loss timeout, small camera, motion gating
   thresholds) are still needed.
4. **Phase 4 (MCP)** and onward as planned below.

**Why Ollama, not vLLM.** The 16 GB of VRAM is shared with Gazebo rendering (and
later YOLO). vLLM reserves a fixed fraction of VRAM up front and is tuned for
throughput across many requests; this stack sends one request at a time. Ollama loads
4-bit GGUF models, sizes its memory to the model and context, and has an
OpenAI-compatible API, so the adapter is the same either way. vLLM can be swapped in
later by pointing `llm.base_url` at it (`llm.provider: vllm`).

**Status (2026-11-16), measured on the VM:**

* Host: Ubuntu 22.04, Docker 29 (`docker.io`) + `docker-compose-v2` + NVIDIA Container
  Toolkit; `check_host.sh --gpu-test` passes.
* Gazebo: PX4's own launch rendered over GLX on `$DISPLAY` and failed (`NV-GLX
  BadValue`: the AnyDesk desktop is not on the NVIDIA GPU), leaving no camera and no
  PX4 data. The sim entrypoint now starts the server with `--headless-rendering` and
  PX4 attaches (`PX4_GZ_STANDALONE=1`); the GUI uses Mesa. The toolkit mounts
  `libEGL_nvidia` but no glvnd vendor file, so the server first rendered on llvmpipe
  (camera ~4.5 Hz of 10 Hz); `compose.gpu.yaml` now mounts `10_nvidia.json`
  (GPU rendering not yet confirmed).
* LLM: `qwen3:14b` fully on the GPU (41/41 layers, ~10.6 GB with 16k context, q8 KV).
  `llm_check` PASS: 1.9 s per request warm; the first load took ~150 s, so the `llm`
  service now preloads the model and only reports healthy once it is in memory.
* Phase 1 re-check passed: takeoff to 5 m, `NavigateTo` (3, 4, 5) succeeded at 0.47 m
  (inside the 0.5 m radius), RTL accepted; no failsafes.
* Phase 2 re-check passed: hovering at (3, 4, 5) facing east, survivor placed at
  (7.68, 3.91), 0.33 m from its SDF position. Fire placed 1.5 m off (10.88, 4.49 vs
  12.0, 5.5): ~8 m away near the image edge, depth sees its near surface.
* Phase 3 run 1 (qwen3:14b, thinking on): flew takeoff -> scan -> move -> scan ->
  survivor, ending 0.44 m above it, but reported failure. Causes: the scan tool marked
  every object unconfirmed (nodes were compacted twice, losing `state:confirmed`), so
  the model re-scanned from above the survivor, where the camera cannot see it. Also
  25-40 s per step from Qwen3 thinking, and the landing pad split into ~7 tracks.
  Fixed: scan keeps `confirmed`; prompt says targets below the vehicle are out of view
  and arriving above one achieves "hover above"; `llm.reasoning_effort: none`
  (thinking off, its text kept in the trace when on); size-aware association gate.
* Phase 3 run 2 (thinking off): all objects now confirmed, ~2 s per step, but no
  plan: it reached the survivor 5 times (0.2-0.5 m) without finishing, flew to the
  landing pad ~15 times, and ran out of its 40 steps. Thinking is back on, and a
  mission sighting log (`reasoning/sightings.py`) now adds `known_objects` (confirmed
  sightings merged across scans, kept when out of view) and, after navigate_to,
  `above` (known objects within 1.5 m) to tool results.
* Context budget: checked and **not** the cause of run 2. Ollama's log
  (`docker compose logs llm | grep -i truncat`) shows every request of runs 1 and 2 with
  `truncated = 0`; run 2's prompt grew ~200 tokens per step to 7.9k at step 40, half of
  the 16k context. Run 2 failed on planning (thinking off), not lost history. The
  safeguards added while checking stay: the goal is also in the system prompt; only the
  newest `known_objects` stays in the history and `scene` / scan `objects` older than 2
  steps are replaced by a note (`reasoning/history.py`; the saved trace keeps
  everything); thinking text goes to the trace only. Each step logs the prompt token
  count and an estimate, warns at 85% of `LLM_CONTEXT`, and the trace records the
  largest prompt (`stats`).
* Next: repeat the mission ~10 times and record the success rate and the largest prompt.

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
