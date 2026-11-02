#!/usr/bin/env bash
gz service -s /world/search_and_rescue_harmonic/set_pose \
  --reqtype gz.msgs.Pose \
  --reptype gz.msgs.Boolean \
  --timeout 2000 \
  --req 'name: "survivor_target", position: {x: 8.0, y: 4.0, z: 0.0}, orientation: {w: 1.0}' 2>/dev/null || true

for i in 0 1 2 3; do
  gz service -s /world/search_and_rescue_harmonic/set_pose \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 1000 \
    --req "name: \"survivor_target_${i}\", position: {x: 14.0, y: 6.0, z: 0.0}, orientation: {w: 1.0}" 2>/dev/null || true
done
echo "Survivors reset to field positions."
