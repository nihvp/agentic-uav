#!/usr/bin/env bash
# Live demo of the paper-reproduction pipeline on one scene, end to end:
# spawn the people -> fly to the viewpoint and capture -> show the frame ->
# vision model's verdict -> return to launch.
#   scripts/demo_scene.sh          scene 1 (an emergency scene)
#   scripts/demo_scene.sh 2        scene 2 (a normal scene)
# The stack must be running (scripts/run.sh). Everything goes to
# experiments/paper/demo/, never to the recorded captures/ or results/, so a demo
# cannot change the reported numbers.
set -uo pipefail
cd "$(dirname "$0")/.."

SCENE=${1:-1}
DEMO=experiments/paper/demo
MODEL=${LLM_MODEL:-qwen3-vl:8b}

exec_agent() {
  docker compose exec -T agent bash -c "source /ws/install/setup.bash && $*"
}
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

rm -rf "$DEMO" && mkdir -p "$DEMO"

step "1/5 Spawn scene $SCENE in Gazebo"
docker compose exec -T sim python3 -m agentic_uav.experiment.spawn --scene "$SCENE" || exit 1

step "2/5 Fly to the viewpoint and capture (action layer: takeoff, navigate, steady hover)"
exec_agent "python3 -m agentic_uav.experiment.capture --scene $SCENE --out /experiments/paper/demo/captures" || exit 1

FRAME=$(printf "%s/captures/scene_%02d/rgb_0.png" "$DEMO" "$SCENE")
step "3/5 The frame the systems see: $FRAME"
(xdg-open "$FRAME" >/dev/null 2>&1 &)

step "4/5 Vision model ($MODEL) analyses the frame with YOLO's detections (thinks first: 10-40 s)"
exec_agent "python3 -m agentic_uav.experiment.analyse --system agentic \
  --captures /experiments/paper/demo/captures --results /experiments/paper/demo/results --only $SCENE" || exit 1

python3 - "$DEMO/results" <<'EOF'
import glob, json, sys
path = sorted(glob.glob(sys.argv[1] + "/agentic_*.jsonl"))[0]
rec = [json.loads(l) for l in open(path) if l.strip()][-1]
o = rec["output"]
print(f"\nYOLO: {len(rec['yolo']['detections'])} persons in {rec['yolo']['time_s'] * 1000:.0f} ms; "
      f"model: {rec['llm']['time_s']:.1f} s, {rec['llm'].get('completion_tokens')} tokens")
print(json.dumps(o, indent=2))
EOF

step "5/5 Return to launch"
exec_agent "ros2 service call /action/flight_command agentic_uav_interfaces/srv/FlightCommand '{command: RTL}'" >/dev/null \
  && echo "RTL sent."
