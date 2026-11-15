# Run guide (Fedora VM)

Day-to-day steps to start the stack, fly by hand, run an LLM mission, and fix the
problems we have already hit. Every command runs **on the Fedora VM** unless it says Mac.

- Repo on the VM: `/mnt/utm-share/agentic-uav`
- Same folder on the Mac: `~/UTM-shared/UTM-Fedora/agentic-uav`

> **Editing files:** edit on the **Mac** side (any editor) — the VM sees the change
> immediately. `sed -i` fails on the VM's shared folder (it cannot create temp files).

---

## 0. What runs where

| Container | What's inside | Started by |
|-----------|---------------|------------|
| `sim`   | PX4 SITL + Gazebo Harmonic + XRCE-DDS agent | `scripts/run_on_fedora.sh` |
| `agent` | ROS 2 Jazzy nodes: action, perception, reasoning (LLM), learning, Gazebo bridge | same script |

The `agent` container rebuilds the ROS workspace on every start (`REBUILD_ON_START=1`),
so code and `params.yaml` changes only need `docker compose restart agent`.
No image rebuild is needed unless a Dockerfile or `agent-requirements.txt` changes.

World `search_and_rescue` (map frame: metres, x east, y north, z up, home = 0,0):

| Object | Position (x, y) | Colour |
|--------|-----------------|--------|
| Survivor | (8.0, 4.0) | orange |
| Fire | (12.0, 5.5) | red |
| Landing pad | (0.0, 8.0) | blue |
| Debris | (1.5, 8.5) | — (not detected) |

Use these to judge whether perception/mission results are right (expect ~0.5 m error).

---

## 1. Start of day

Terminal **A** (stack, leave it running):

```bash
cd /mnt/utm-share/agentic-uav
```

```bash
free -h
```
Expect about 4.8 Gi total. Close other apps in the VM if "available" is under ~3 Gi.

```bash
export GEMINI_API_KEY=your-key-here
```

Pick one start command:

| Goal | Command |
|------|---------|
| **LLM mission (normal)** | `AGENT_LAUNCH_ARGS="learning:=false" scripts/run_on_fedora.sh --headless` |
| Flight + perception only, no LLM | `AGENT_LAUNCH_ARGS="reasoning:=false learning:=false" scripts/run_on_fedora.sh --headless` |
| Flight only | `AGENT_LAUNCH_ARGS="perception:=false reasoning:=false learning:=false" scripts/run_on_fedora.sh --headless` |
| With the Gazebo window (heavier) | drop `--headless` |
| After a Dockerfile change | add `--build` |

`--headless` saves a lot of CPU/RAM; use it unless you need to watch.
The key must be exported **in terminal A before starting**. If you forgot, press
Ctrl+C, export it, and start again (`docker compose restart` does not pick up new env vars).

Wait ~1–2 minutes. Ready when terminal A shows lines like `Action layer up`,
`Perception layer up`, and the reasoning node started. Early `Waiting for: /camera/image...`
or `No valid PX4 local position` warnings are normal for the first ~30–60 s.

---

## 2. Open a ROS shell

Terminal **B** (and C for logs):

```bash
cd /mnt/utm-share/agentic-uav
```

```bash
docker compose exec agent bash
```

Inside the container:

```bash
source /ws/install/setup.bash
```

All `ros2 ...` commands below run inside this shell.
("Could not determine the type" right after starting is DDS discovery lag — wait 5 s, retry.)

### Quick health check

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
About 2–3 Hz on this VM is normal (perception only).

---

## 3. Fly by hand (no LLM)

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

## 4. Run the LLM mission

The drone should be **on the ground** (`armed: false`) — the agent takes off itself.

Terminal **C** — watch the agent think (outside the container, in the repo folder):

```bash
docker compose logs -f agent | grep reasoning_node
```

Terminal **B** — start the mission (this call blocks until the mission ends):

```bash
ros2 service call /reasoning/run_mission agentic_uav_interfaces/srv/RunMission "{goal: 'Find the survivor and hover above them.'}"
```

In terminal C you'll see, per step: `asking the LLM for the next action...`,
`calling <tool>(args) | <thought>`, then the result. Expected flow: takeoff → scan →
(move ~10 m, scan again) → navigate to the survivor at ~(8, 4, 5) → finish_mission.

**Success** = the response says `success: True` and telemetry is close to x 8, y 4, z 5.

After the mission, read the full trace (outside the container):

```bash
docker compose exec agent bash -c 'cat $(ls -t /data/missions/*.json | head -1)'
```

Save a copy to the shared folder (so you can open it on the Mac):

```bash
docker compose exec agent bash -c 'cat $(ls -t /data/missions/*.json | head -1)' > mission_latest.json
```

Then RTL (section 3) before starting another mission.

---

## 5. LLM quota and switching models

Gemini free tier is about **5 requests/min and ~20 requests/day per model**. One mission
uses roughly 8–20 requests, so you get about **1–2 missions per model per day**.
The daily quota resets at **midnight US Pacific = 12:00 noon UAE time (UTC+4)**.
Check usage: https://ai.dev/rate-limit

The model is set in `src/agentic_uav_bringup/config/params.yaml` (edit on the Mac):

```yaml
llm.model: gemini-3.5-flash
```

| Model | Notes |
|-------|-------|
| `gemini-3.5-flash` | Default. 3–15 s per request. Separate daily quota. |
| `gemini-3.5-flash-lite` | Faster, separate quota; a bit less reliable at multi-step planning. Good backup. |
| `gemini-3.8-flash` | **Don't use**: 80–90+ s per request on free tier → 504 errors. |
| `gemini-2.5-*` | **Don't use**: 404, closed to new users. |

Each model has its own quota, so when one is exhausted switch to the other.
After editing, land first, then:

```bash
docker compose restart agent
```

---

## 6. Gazebo window (to watch the drone)

Either start without `--headless`, or attach a viewer to a headless sim:

```bash
xhost +si:localuser:$(whoami)
```

```bash
docker compose exec sim gz sim -g
```

Close the window when done; the simulation keeps running.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `GEMINI_API_KEY is not set` | Export it in terminal A, Ctrl+C, start again. |
| `429 RESOURCE_EXHAUSTED` / `Gemini quota exhausted ... wait N s` | Per-minute limit: it waits and retries by itself. Daily limit (wait > 60 s): mission stops — switch model (section 5) or wait for noon UTC+4. |
| `404 ... no longer available to new users` | Wrong model name — use one from the table in section 5. |
| `504 DEADLINE_EXCEEDED` | Model too slow (3.8-flash). Switch to 3.5-flash. |
| `Cannot navigate in state TAKING_OFF` | Takeoff still in progress; wait for `HOLDING`. |
| `No valid PX4 local position yet` | PX4 still initialising. Wait 30–60 s and retry. |
| Drone lands by itself mid-flight | PX4 failsafe from a VM stall. Check `docker compose logs sim \| grep -iE "failsafe\|offboard\|landing\|disarm" \| tail -20`. Telemetry `last_event` also says why. Free RAM, use `--headless`. |
| `controller_state: EXTERNAL` | PX4 took control (failsafe/RTL). Send TAKEOFF again to resume in place, or RTL. |
| Perception tracks nothing | `docker compose logs agent \| grep "Frames used" \| tail -3`. If "skipped mid-turn" keeps rising while hovering, report it. Check camera with `ros2 topic hz /camera/image`. |
| No `/camera/*` topics | Bridge not up: `docker compose logs agent \| grep -iE "bridge\|camera" \| head -20`. |
| Mission seems stuck | Watch terminal C; a single LLM request can take ~15 s, plus pacing waits of up to 12 s between requests on the free tier. |
| Simulation slow / "time jump detected" | VM under load. Close other apps; `free -h`; stay headless. |
| Disk full | `docker image prune -f`; `df -h /`. |

Useful log commands (outside the container):

```bash
docker compose logs agent | grep action_node | tail -20
```

```bash
docker compose logs sim --since 2m | grep -E "WARN|ERROR" | tail -20
```

```bash
docker compose ps -a
```

---

## 8. Never restart the agent mid-flight

Restarting `agent` while flying stops offboard setpoints, which triggers the PX4 failsafe.
Always **RTL and wait for `armed: false`** before `docker compose restart agent` or Ctrl+C.

---

## 9. End of day

1. RTL (section 3) and wait for `armed: false`.
2. Ctrl+C in terminal A, then:

```bash
docker compose down
```

Mission traces survive in the `agent-data` volume.

---

## 10. Unit tests (optional, Mac or VM, no simulator needed)

```bash
cd src/agentic_uav && python3 -m pytest test -p no:cacheprovider
```

Needs `numpy`, `opencv-python-headless`, `pytest`, `google-genai` installed.
