"""
Default Standard Operating Procedures (SOPs) and Cross-Mission Memory Knowledge Base.
Provides verified protocols for UAV Search & Rescue, Fire Reconnaissance,
Obstacle Management, and Emergency Landing Site Evaluation.
"""

from typing import List, Dict, Any

DEFAULT_KNOWLEDGE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "SOP-SAR-001",
        "category": "SOP",
        "title": "Search and Rescue Victim Confirmation Protocol",
        "content": (
            "When a human survivor is detected by the perception layer: 1) Immediately record 3D GPS coordinates. "
            "2) Transition UAV flight mode to LOITER at a minimum safe altitude of 5.0m to avoid rotor downwash injury. "
            "3) Direct camera gimbal to track the survivor. 4) Check for secondary hazards (fire, debris, flood) in the "
            "immediate 15m radius. 5) Dispatch critical emergency alert via MCP Emergency Dispatch service with precise coordinates."
        ),
        "tags": ["sar", "survivor", "inspection", "protocol"]
    },
    {
        "id": "SOP-HAZ-002",
        "category": "SOP",
        "title": "Wildfire and Thermal Hazard Containment Protocol",
        "content": (
            "Upon detecting active fire hazard or severe thermal anomaly: 1) Maintain minimum standoff distance of 12.0m. "
            "2) Evaluate wind direction to avoid flying downwind through convective smoke plume. 3) Cross-reference "
            "scene graph for nearby victims or critical infrastructure. 4) Notify UTM and Emergency Operations Center "
            "with fire front heading and perimeter coordinates."
        ),
        "tags": ["fire", "hazard", "thermal", "standoff"]
    },
    {
        "id": "SOP-LZ-003",
        "category": "SOP",
        "title": "Emergency Landing Zone Clearance and Obstacle Evaluation",
        "content": (
            "For emergency landing or casualty medevac landing zone verification: 1) Verify a minimum 4.0m x 4.0m flat surface. "
            "2) Ensure slope does not exceed 8 degrees. 3) If debris or obstacles are within 3.0m of the landing pad center, "
            "flag the landing zone as 'BLOCKED' and query alternative landing coordinates from regional map. "
            "4) Confirm ground wind velocity is under 10 m/s before executing descent."
        ),
        "tags": ["landing_zone", "clearance", "debris", "obstacle"]
    },
    {
        "id": "REG-FAA-004",
        "category": "REGULATION",
        "title": "FAA Part 107 / UTM Geofence Compliance Rules",
        "content": (
            "UAV must operate strictly within predefined geofence ceiling of 25.0m AGL and maximum radius of 150.0m "
            "from launch point unless authorized by UTM clearance. If battery level drops below 20%, an automated "
            "Return-to-Launch (RTL) failsafe must be executed immediately without human override delay."
        ),
        "tags": ["faa", "geofence", "utm", "battery", "rtl"]
    },
    {
        "id": "MEM-MISSION-2025-09",
        "category": "MISSION_LOG",
        "title": "Mission Bravo After-Action Review: Industrial Sector Debris",
        "content": (
            "Historical Mission Report: In Sector Bravo, fallen scaffolding and metal debris frequently occlude ground "
            "sensors from high angles. Optimal detection was achieved by descending to 4.5m and pitching gimbal to 60 degrees. "
            "Secondary safe landing zone was identified at coordinates [18.2, -4.5, 0.0]."
        ),
        "tags": ["mission_log", "lessons_learned", "sector_bravo", "camera_pitch"]
    }
]
