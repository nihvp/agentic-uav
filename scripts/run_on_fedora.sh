#!/usr/bin/env bash
# Prepare the Fedora VM host and start the stack.
#   scripts/run_on_fedora.sh [--build] [--headless] [--software-gl] [--sim-only]
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="" ; SERVICES=(sim agent)
for arg in "$@"; do
  case "$arg" in
    --build) BUILD="--build" ;;
    --headless) export HEADLESS=1 ;;
    --software-gl) export LIBGL_ALWAYS_SOFTWARE=1 ;;
    --sim-only) SERVICES=(sim) ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "Install Docker first: sudo dnf install docker-ce docker-compose-plugin (or moby-engine)"; exit 1; }

if [[ -z "${HEADLESS:-}" ]] && command -v xhost >/dev/null; then
  xhost +si:localuser:"$(whoami)" >/dev/null || true
fi

FILES=(-f compose.yaml)
if [[ -e /dev/dri ]]; then
  FILES+=(-f compose.gpu.yaml)
  card=$(ls /dev/dri/card* 2>/dev/null | head -1)
  render=$(ls /dev/dri/renderD* 2>/dev/null | head -1)
  export DRI_CARD_GID=$(stat -c %g "${card:-/dev/dri}")
  export DRI_RENDER_GID=$(stat -c %g "${render:-${card:-/dev/dri}}")
else
  echo "No /dev/dri: using software rendering."
  export LIBGL_ALWAYS_SOFTWARE=1
fi

[[ -n "${GEMINI_API_KEY:-}" ]] || echo "Note: GEMINI_API_KEY is not set; reasoning_node will fail to start."

docker compose "${FILES[@]}" up $BUILD "${SERVICES[@]}"
