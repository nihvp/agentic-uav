#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /opt/px4_ws/install/setup.bash
# With ./src mounted for development, rebuild so new files and interfaces are picked up.
if [[ "${REBUILD_ON_START:-0}" == "1" ]]; then
  (cd /ws && colcon build --symlink-install --event-handlers console_direct-)
fi
source /ws/install/setup.bash
exec "$@"
