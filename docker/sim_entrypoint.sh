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
GZ_PID="" ; GUI_PID=""
trap 'kill ${AGENT_PID} ${PARAMS_PID} ${GZ_PID} ${GUI_PID} 2>/dev/null || true' EXIT

# Start Gazebo here instead of letting PX4 do it, so the server renders the camera with
# --headless-rendering (EGL straight on the GPU). PX4's own launch renders over GLX on
# $DISPLAY, which fails when the desktop is not driven by the NVIDIA GPU (remote VM:
# "X Error ... NV-GLX BadValue", then no camera and a stalled lockstep clock).
cd "${PX4_DIR}"
GZ_ENV="${PX4_DIR}/build/px4_sitl_default/rootfs/gz_env.sh"   # model/world/plugin paths
[[ -f "${GZ_ENV}" ]] || { echo "[sim_entrypoint] missing ${GZ_ENV}; rebuild the sim image"; exit 1; }
set +u   # gz_env.sh appends to possibly unset GZ_SIM_* variables
# shellcheck disable=SC1090
source "${GZ_ENV}"
set -u
gz sim --verbose="${GZ_VERBOSE:-1}" -r -s --headless-rendering "${PX4_GZ_WORLDS}/${WORLD}.sdf" &
GZ_PID=$!
for _ in $(seq 120); do
  # No pipe into grep -q: with pipefail an early grep exit fails the check.
  [[ "$(gz service -i --service "/world/${WORLD}/scene/info" 2>&1)" == *"Service providers"* ]] && break
  kill -0 "${GZ_PID}" 2>/dev/null || { echo "[sim_entrypoint] Gazebo server exited"; exit 1; }
  sleep 1
done
echo "[sim_entrypoint] Gazebo server up: world ${WORLD}, camera rendered headless (EGL)"

if [[ -z "${HEADLESS:-}" ]]; then
  # The GUI only draws the view, so Mesa software rendering is fine and works on any X
  # server. GZ_GUI_GPU=1 tries the NVIDIA GLX driver instead.
  (
    if [[ "${GZ_GUI_GPU:-0}" != 1 ]]; then export __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1; fi
    gz sim -g 2>&1 | sed -u 's/^/[gz gui] /'
  ) &
  GUI_PID=$!
fi

# PX4 connects to the running world and spawns the vehicle into it.
export PX4_GZ_WORLD="${WORLD}" PX4_GZ_STANDALONE=1
make px4_sitl "${MODEL}"
