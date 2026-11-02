#!/usr/bin/env python3
"""
Diagnostic utility to inspect all entities and positions in the Gazebo simulation.
Run inside container: python3 /ros2_ws/src/show_world.py
"""

import re
import subprocess
import numpy as np


def main():
    print("=== Scanning Gazebo World Entities ===")
    models = {}

    # Method 1: Try reading gz topic pose/info
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", "/world/search_and_rescue_harmonic/pose/info", "-n", "1"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True
        )
        for block in out.split("pose {"):
            name_m = re.search(r'name:\s*"([^"]+)"', block)
            if not name_m:
                continue
            name = name_m.group(1)
            pos_m = re.search(r'position\s*\{\s*x:\s*([-\d\.]+)\s*y:\s*([-\d\.]+)\s*z:\s*([-\d\.]+)', block)
            if pos_m:
                models[name] = np.array([float(pos_m.group(1)), float(pos_m.group(2)), float(pos_m.group(3))])
    except Exception:
        pass

    # Method 2: Try gz model --list fallback
    if not models:
        try:
            l_out = subprocess.check_output(
                ["gz", "model", "--list", "-w", "search_and_rescue_harmonic"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                text=True
            )
            for line in l_out.splitlines():
                name = line.strip().lstrip("-").strip()
                if not name:
                    continue
                try:
                    p_out = subprocess.check_output(
                        ["gz", "model", "-m", name, "-p", "-w", "search_and_rescue_harmonic"],
                        stderr=subprocess.DEVNULL,
                        timeout=1.0,
                        text=True
                    )
                    m = re.search(r'\[\s*([-\d\.]+)[,\s]+([-\d\.]+)[,\s]+([-\d\.]+)\]', p_out)
                    if m:
                        models[name] = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                except Exception:
                    pass
        except Exception:
            pass

    if not models:
        print("[WARN] Could not retrieve Gazebo models via CLI. Simulation might still be initializing.")
        return

    print(f"Found {len(models)} active entities in world:\n")
    for name, pos in sorted(models.items()):
        status = ""
        name_l = name.lower()
        if "survivor" in name_l:
            d_pad = np.linalg.norm(pos[:2] - np.array([0.0, 8.0]))
            if d_pad < 2.2:
                status = " [RESCUED at Landing Pad]"
            else:
                status = f" [UNRESCUED in field at ({pos[0]:.1f}, {pos[1]:.1f})]"
        elif "fire" in name_l:
            status = f" [HAZARD at ({pos[0]:.1f}, {pos[1]:.1f})]"
        elif name == "agentic_uav":
            status = f" [UAV POSE at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})]"

        print(f" - {name:22s} -> position: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}){status}")


if __name__ == "__main__":
    main()
