#!/usr/bin/env bash
# Starts the uXRCE-DDS agent and PX4 SITL with our world.
set -euo pipefail

WORLD="${PX4_GZ_WORLD:-search_and_rescue}"
MODEL="${PX4_SIM_MODEL:-gz_x500_depth}"
PX4_DIR=/opt/PX4-Autopilot

# PX4 looks up worlds by name in its own worlds directory.
if [[ -f "/worlds/${WORLD}.sdf" ]]; then
  ln -sf "/worlds/${WORLD}.sdf" "${PX4_DIR}/Tools/simulation/gz/worlds/${WORLD}.sdf"
fi

MicroXRCEAgent udp4 -p 8888 &
AGENT_PID=$!
trap 'kill ${AGENT_PID} 2>/dev/null || true' EXIT

export PX4_GZ_WORLD="${WORLD}"
cd "${PX4_DIR}"
make px4_sitl "${MODEL}"
