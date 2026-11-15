#!/usr/bin/env bash
# Start the stack on the Ubuntu GPU VM.
#   scripts/run.sh [--build] [--headless] [--software-gl] [--sim-only] [--no-llm]
# Run scripts/check_host.sh first on a new machine.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="" ; SERVICES=(sim agent llm)
for arg in "$@"; do
  case "$arg" in
    --build) BUILD="--build" ;;
    --headless) export HEADLESS=1 ;;
    --software-gl) export LIBGL_ALWAYS_SOFTWARE=1 ;;
    --sim-only) SERVICES=(sim) ;;
    --no-llm) SERVICES=(sim agent) ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "Docker is not installed; see docs/RUN_GUIDE.md section 1."; exit 1; }

# Gazebo GUI on the desktop session (AnyDesk shows the VM's X11 display).
if [[ -z "${HEADLESS:-}" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set (SSH session?). Use --headless or run from the desktop terminal." >&2; exit 1
  fi
  if command -v xhost >/dev/null; then xhost +si:localuser:"$(whoami)" >/dev/null || true; fi
fi

FILES=(-f compose.yaml)
if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
  FILES+=(-f compose.gpu.yaml)
else
  echo "Docker has no NVIDIA runtime: running on CPU with software GL (see scripts/check_host.sh)."
  export LIBGL_ALWAYS_SOFTWARE=1
fi

docker compose "${FILES[@]}" up $BUILD "${SERVICES[@]}"
