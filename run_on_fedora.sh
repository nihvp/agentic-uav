#!/usr/bin/env bash
# ==============================================================================
# Helper deployment script for Fedora VM Host
# Configures Wayland/X11 access, GPU DRI acceleration, and launches Docker
# ==============================================================================

set -e

echo "=== [Agentic UAV] Preparing Fedora VM Host Environment ==="

# 1. Authorize local container connections to X11 server
if command -v xhost &> /dev/null; then
    echo "Configuring X11 display authorization..."
    xhost +local:root || xhost +si:localuser:$(whoami) || true
fi

# 2. Check for DRI hardware acceleration device
if [ -d "/dev/dri" ]; then
    echo "Detected DRI graphics device (/dev/dri). Hardware OpenGL acceleration enabled."
else
    echo "Warning: /dev/dri not found. Software rasterizer (llvmpipe) will be utilized."
fi

# 3. Export current host DISPLAY if unset
export DISPLAY="${DISPLAY:-:0}"

# 4. Check GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Notice: GEMINI_API_KEY is not set in environment."
    echo "The Reasoning Layer will run in self-contained deterministic ReAct mode."
    echo "To activate live Gemini multimodal reasoning, run: export GEMINI_API_KEY='your_key'"
fi

# 5. Build and launch container cleanly
echo "Stopping any previous container instances..."
docker compose down --remove-orphans || true

echo "Building and launching container via docker compose..."
docker compose build
docker compose up --force-recreate
