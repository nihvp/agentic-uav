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

# Camera mount and resolution. Patched from pristine copies so restarts don't stack edits.
MODELS="${PX4_DIR}/Tools/simulation/gz/models"
for f in x500_depth/model.sdf OakD-Lite/model.sdf; do
  [[ -f "${MODELS}/${f}.orig" ]] || cp "${MODELS}/${f}" "${MODELS}/${f}.orig"
done
sed "s|<pose\(.*\)>.12 .03 .242 0 0 0</pose>|<pose\1>.12 .03 .242 0 ${CAMERA_PITCH_RAD:-0} 0</pose>|" \
  "${MODELS}/x500_depth/model.sdf.orig" > "${MODELS}/x500_depth/model.sdf"
sed -e "s|<width>1920</width>|<width>${CAMERA_WIDTH:-1920}</width>|" \
    -e "s|<height>1080</height>|<height>${CAMERA_HEIGHT:-1080}</height>|" \
    -e "s|<update_rate>30</update_rate>|<update_rate>${CAMERA_RATE_HZ:-30}</update_rate>|g" \
  "${MODELS}/OakD-Lite/model.sdf.orig" > "${MODELS}/OakD-Lite/model.sdf"
grep -q ".242 0 ${CAMERA_PITCH_RAD:-0} 0</pose>" "${MODELS}/x500_depth/model.sdf" \
  || echo "[sim_entrypoint] WARNING: camera mount pose not found in x500_depth/model.sdf; pitch not applied"
echo "[sim_entrypoint] camera pitch ${CAMERA_PITCH_RAD:-0} rad, RGB ${CAMERA_WIDTH:-1920}x${CAMERA_HEIGHT:-1080} @ ${CAMERA_RATE_HZ:-30} Hz"

# Apply PX4_PARAM_<NAME>=<value> env vars once PX4 is running. Setting them during
# startup does not stick: the SITL scripts set their own defaults afterwards.
apply_params() {
  local bin="${PX4_DIR}/build/px4_sitl_default/bin"
  for _ in $(seq 600); do
    "${bin}/px4-param" show SYS_AUTOSTART >/dev/null 2>&1 && break
    sleep 1
  done
  sleep 5
  env | grep '^PX4_PARAM_' | sort | while IFS='=' read -r key value; do
    echo "[sim_entrypoint] param set ${key#PX4_PARAM_} ${value}"
    "${bin}/px4-param" set "${key#PX4_PARAM_}" "${value}" || true
  done
}

MicroXRCEAgent udp4 -p 8888 &
AGENT_PID=$!
apply_params &
PARAMS_PID=$!
trap 'kill ${AGENT_PID} ${PARAMS_PID} 2>/dev/null || true' EXIT

export PX4_GZ_WORLD="${WORLD}"
cd "${PX4_DIR}"
make px4_sitl "${MODEL}"
