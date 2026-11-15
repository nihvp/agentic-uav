# Run guide (remote Ubuntu GPU VM)

Day-to-day steps to set up the VM, start the stack, fly by hand, run an LLM mission,
and fix known problems. Commands run **on the VM** (AnyDesk desktop terminal) unless
they say **Mac**.

- Repo on the VM: `~/Desktop/AgenticAI_Drones/agentic-uav` (a git clone, branch `rebuild`)
- Repo on the Mac: `~/UTM-shared/UTM-Fedora/agentic-uav`
- VM: x86_64, 8 CPUs, 53 GiB RAM, RTX 5070 Ti (16 GB), Ubuntu with an X11 desktop

---

## 0. What runs where

| Container | What's inside | GPU use |
|-----------|---------------|---------|
| `sim`   | PX4 SITL + Gazebo Harmonic + XRCE-DDS agent | Gazebo rendering (GUI + depth camera) |
| `agent` | ROS 2 Jazzy nodes: action, perception, reasoning, learning, Gazebo bridge | torch (YOLO, embeddings) |
| `llm`   | Ollama serving `LLM_MODEL` (OpenAI-compatible API on `127.0.0.1:11434/v1`) | model weights + KV cache |

All three use host networking. The 16 GB of VRAM is shared: Gazebo takes ~1-2 GB,
the default model (`qwen3:14b`, 4-bit) ~10-11 GB with its 16k context.

The `agent` container rebuilds the ROS workspace on every start (`REBUILD_ON_START=1`),
so code and `params.yaml` changes only need `docker compose restart agent`. Rebuild the
image (`--build`) only when a Dockerfile or `docker/agent-requirements.txt` changes.

World `search_and_rescue` (map frame: metres, x east, y north, z up, home = 0,0):

| Object | Position (x, y) | Colour |
|--------|-----------------|--------|
| Survivor | (8.0, 4.0) | orange |
| Fire | (12.0, 5.5) | red |
| Landing pad | (0.0, 8.0) | blue |
| Debris | (1.5, 8.5) | — (not detected) |

Use these to judge perception and mission results (expect ~0.5 m error).

---

## 1. One-time VM setup

1. Docker Engine + compose plugin: follow https://docs.docker.com/engine/install/ubuntu/
   then let your user run it without sudo:

   ```bash
   sudo usermod -aG docker $USER
   ```
   Log out and back in (or reboot) for the group to apply.

2. NVIDIA Container Toolkit: install `nvidia-container-toolkit` following
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   then register it with Docker:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   ```

3. Clone the repo:

   ```bash
   cd ~/Desktop/AgenticAI_Drones && git clone -b rebuild https://github.com/nihvp/agentic-uav.git
   ```

4. Settings file (git-ignored; compose reads it automatically):

   ```bash
   cd ~/Desktop/AgenticAI_Drones/agentic-uav && cp .env.example .env
   ```

5. Check the host. Fix every FAIL before going on:

   ```bash
   scripts/check_host.sh --gpu-test
   ```

6. First build and start. The PX4 build and the model download (~9 GB) take a while:

   ```bash
   scripts/run.sh --build
   ```

---

## 2. Daily workflow (Mac ↔ VM)

1. **Mac:** edit, commit, `git push`.
2. **VM:** pull (stack can keep running if the drone is landed):

   ```bash
   cd ~/Desktop/AgenticAI_Drones/agentic-uav && git pull
   ```
3. **VM:** apply it. Python or `params.yaml` change:

   ```bash
   docker compose restart agent
   ```
   Dockerfile, requirements or `compose*.yaml` change: Ctrl+C the stack and start it
   again with `--build`.
4. Run and test (sections 4-6), copy findings or mission traces back to the Mac
   (commit them under `docs/` or paste them into the chat).

---

## 3. Start of day

Terminal **A** (stack, leave it running):

```bash
cd ~/Desktop/AgenticAI_Drones/agentic-uav && git pull
```

| Goal | Command |
|------|---------|
| **Everything, Gazebo window on the desktop** | `scripts/run.sh` |
| Everything, no Gazebo window | `scripts/run.sh --headless` |
| No LLM (flight + perception only) | `AGENT_LAUNCH_ARGS="reasoning:=false" scripts/run.sh --no-llm` |
| Flight only | `AGENT_LAUNCH_ARGS="perception:=false reasoning:=false learning:=false" scripts/run.sh --no-llm` |
| After a Dockerfile / requirements change | add `--build` |

Ready when terminal A shows `Action layer up`, `Perception layer up`, the reasoning
node started, and `[llm] qwen3:14b ready`. Early `Waiting for: /camera/image...` or
`No valid PX4 local position` warnings are normal for the first ~30 s.

---

## 4. Open a ROS shell and check health

Terminal **B**:

```bash
cd ~/Desktop/AgenticAI_Drones/agentic-uav && docker compose exec agent bash
```

Inside the container:

```bash
source /ws/install/setup.bash
```

All `ros2 ...` and `python3 -m ...` commands below run inside this shell.
("Could not determine the type" right after starting is DDS discovery lag; wait 5 s.)

```bash
ros2 topic echo --once /action/telemetry | grep -E "armed|controller_state|last_event"
```
Expect `controller_state: IDLE`, `armed: false`.

```bash
ros2 topic echo --once /fmu/out/vehicle_local_position | grep -E "^(xy_valid|z_valid):"
```
Both `true` = PX4 is ready to fly.

```bash
ros2 topic hz /camera/image --window 20
```
Note the rate (the Fedora VM managed 2-3 Hz) and write it in REBUILD_PLAN.md.

LLM check: one request with one tool, must print `PASS`. The first call loads the
model into VRAM (up to a minute); later calls take a few seconds.

```bash
python3 -m agentic_uav.reasoning.llm_check
```

GPU memory (on the VM, outside the container):

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

---

## 5. Fly by hand (no LLM)

Take off to 5 m:

```bash
ros2 service call /action/flight_command agentic_uav_interfaces/srv/FlightCommand "{command: TAKEOFF, altitude: 5.0}"
```

Wait until `controller_state: HOLDING` (call takeoff only **once**):

```bash
ros2 topic echo --once /action/telemetry | grep -E "controller_state|z:"
```

Fly to a point facing a heading (`yaw` in radians, ENU: 0 = east, 1.57 = north, 3.14 = west; `.nan` = keep heading):

```bash
ros2 action send_goal --feedback /action/navigate_to agentic_uav_interfaces/action/NavigateTo "{target: {x: 3.0, y: 4.0, z: 5.0}, yaw: 0.0, acceptance_radius: 0.0}"
```

See what perception tracks:

```bash
ros2 topic echo --once /perception/scene_graph | grep -E "^  label|^    [xyz]:" | paste - - - -
```

Come home and land (always do this before stopping anything):

```bash
ros2 service call /action/flight_command agentic_uav_interfaces/srv/FlightCommand "{command: RTL, altitude: 0.0}"
```

Other commands: `{command: HOLD}`, `{command: LAND, altitude: 0.0}`.

---

## 6. Run the LLM mission

The drone should be **on the ground** (`armed: false`); the agent takes off itself.

Terminal **C**: watch the agent think (on the VM, in the repo folder):

```bash
docker compose logs -f agent | grep reasoning_node
```

Terminal **B**: start the mission (blocks until it ends):

```bash
ros2 service call /reasoning/run_mission agentic_uav_interfaces/srv/RunMission "{goal: 'Find the survivor and hover above them.'}"
```

Per step terminal C shows `asking the LLM for the next action...`,
`calling <tool>(args) | <thought>`, then the result. Expected flow: takeoff → scan →
(move, scan again) → navigate to the survivor at ~(8, 4, 5) → finish_mission.

**Success** = the response says `success: True` and telemetry is close to x 8, y 4, z 5.

Save the trace into the repo so it can be pushed and read on the Mac:

```bash
mkdir -p traces && docker compose exec agent bash -c 'cat $(ls -t /data/missions/*.json | head -1)' > traces/mission_$(date +%Y%m%d_%H%M%S).json
```

Then RTL (section 5) before the next mission. There is no quota now, so run the
mission many times and note the success rate.

---

## 7. Changing the LLM

Set the model in `.env` (on the VM) and restart the stack:

```bash
LLM_MODEL=qwen3:14b
```

The `llm` service pulls it on start; `reasoning_node` asks for the same name. Rules:
the model must support **tool calling** in Ollama (marked "tools" on ollama.com), and
weights + KV cache must leave ~2 GB of VRAM for Gazebo. On 16 GB that means ~14B at
4-bit, or ~8B at 8-bit. `LLM_CONTEXT` (default 16384) costs VRAM too.

List and remove downloaded models:

```bash
docker compose exec llm ollama list
```

```bash
docker compose exec llm ollama rm <model>
```

Gemini still works: set `llm.provider: gemini` and a Gemini `llm.model` in
`params.yaml`, put `GEMINI_API_KEY` in `.env`, and set `llm.requests_per_minute: 5.0`
on the free tier.

---

## 8. Gazebo window

`scripts/run.sh` without `--headless` opens it on the VM desktop (visible in AnyDesk).
To attach a viewer to a headless sim:

```bash
xhost +si:localuser:$(whoami) && docker compose exec sim gz sim -g
```

Closing the window leaves the simulation running.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `check_host.sh`: NVIDIA runtime FAIL | Section 1 step 2; `docker info \| grep -i runtimes` must list `nvidia`. |
| `could not select device driver "nvidia"` | Same as above, then restart Docker. |
| `DISPLAY is not set` | You are in SSH, not the desktop terminal. Use `--headless` or the AnyDesk terminal. |
| Gazebo window does not open / `cannot open display` | `xhost +si:localuser:$(whoami)`; check `echo $DISPLAY` matches the desktop. |
| Gazebo renders on CPU (slow, `llvmpipe` in logs) | GPU overlay not applied: `scripts/run.sh` prints "no NVIDIA runtime" in that case. |
| `llm_check` connection error | `docker compose logs llm \| tail`; still pulling the model, or port 11434 is taken by a host Ollama (`sudo systemctl stop ollama`). |
| `llm_check` FAIL (text, no tool call) | Model does not do tool calls well; try another tools-capable model (section 7). |
| Ollama `out of memory` / model on CPU (slow) | Model + context too big next to Gazebo: smaller model, lower `LLM_CONTEXT`, or `--headless`. |
| CUDA error in agent (`no kernel image`) | Image has old torch; rebuild with `--build` (needs CUDA 12.8 wheels for RTX 50xx). |
| Permission denied writing in `src/` | Host user is not uid 1000; see `check_host.sh`. |
| `Cannot navigate in state TAKING_OFF` | Takeoff still in progress; wait for `HOLDING`. |
| `No valid PX4 local position yet` | PX4 still initialising; wait 30 s. |
| Drone lands by itself / `controller_state: EXTERNAL` | PX4 failsafe. `docker compose logs sim \| grep -iE "failsafe\|offboard\|landing\|disarm" \| tail -20`; telemetry `last_event` says why. Send TAKEOFF to resume, or RTL. |
| Perception tracks nothing | `docker compose logs agent \| grep "Frames used" \| tail -3`; `ros2 topic hz /camera/image`. |
| No `/camera/*` topics | `docker compose logs agent \| grep -iE "bridge\|camera" \| head -20`. |
| Disk full | `docker image prune -f`; `df -h /`. |

---

## 10. Rules

- **Never restart `agent` mid-flight.** It stops offboard setpoints and triggers the
  PX4 failsafe. RTL and wait for `armed: false` first.
- **End of day:** RTL, wait for `armed: false`, Ctrl+C in terminal A, then
  `docker compose down`. Mission traces stay in the `agent-data` volume, models in
  `ollama-models`.

---

## 11. Unit tests (Mac or VM, no simulator)

```bash
cd src/agentic_uav && python3 -m pytest test -p no:cacheprovider
```

Needs `numpy`, `opencv-python-headless`, `pytest`, `openai` (and `google-genai` for the
Gemini tests, which are skipped without it).
