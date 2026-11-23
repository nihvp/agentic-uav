#!/usr/bin/env bash
# Capture the paper-reproduction scenes: for each scene, spawn its people in Gazebo,
# fly to its viewpoint and save frames to experiments/paper/captures/scene_NN/.
#   scripts/capture_scenes.sh            all 44 scenes, then return to launch
#   scripts/capture_scenes.sh 1 2 3      only these scenes
# The stack must be running (scripts/run.sh). The drone takes off at the first scene
# and stays airborne between scenes. Scenes already captured (meta.json present) are
# skipped, so the script can be re-run after a failure; delete a scene's directory to
# capture it again.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/paper/captures
mkdir -p "$OUT"
if [[ $# -gt 0 ]]; then IDS=("$@"); else IDS=($(seq 1 44)); fi

exec_agent() {
  docker compose exec -T agent bash -c "source /ws/install/setup.bash && $*"
}

failed=()
for id in "${IDS[@]}"; do
  dir=$(printf "%s/scene_%02d" "$OUT" "$id")
  if [[ -f "$dir/meta.json" ]]; then echo "[capture_scenes] scene $id already captured, skipping"; continue; fi
  echo "[capture_scenes] ===== scene $id ====="
  if ! docker compose exec -T sim python3 -m agentic_uav.experiment.spawn --scene "$id"; then
    failed+=("$id"); continue
  fi
  if ! exec_agent "python3 -m agentic_uav.experiment.capture --scene $id"; then
    failed+=("$id"); continue
  fi
  # Gazebo's true drone pose, for evaluation only (how far PX4's estimate was off).
  # Analysers must not read this file.
  docker compose exec -T sim gz model -m x500_depth_0 -p > "$dir/eval_true_pose.txt" 2>&1 || true
done

echo "[capture_scenes] returning to launch"
exec_agent "ros2 service call /action/flight_command agentic_uav_interfaces/srv/FlightCommand '{command: RTL}'" >/dev/null || true

done_count=$(ls "$OUT"/scene_*/meta.json 2>/dev/null | wc -l)
echo "[capture_scenes] $done_count scenes captured in $OUT"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "[capture_scenes] failed: ${failed[*]} (re-run the script to retry them)"
  exit 1
fi
