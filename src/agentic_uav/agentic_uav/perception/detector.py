"""Object detectors: image in, 2D boxes out.

ColorDetector is a deliberately simple HSV-threshold detector for the solid-coloured
stand-ins in worlds/search_and_rescue.sdf. It is not object recognition: it finds
"orange blobs", which this world uses for the survivor. Every detection carries
source="color" so downstream layers can tell. It needs no model weights or torch,
which keeps the perception layer small enough for a 4-5 GB VM.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


class Detector:
    """Returns [{"bbox": [x1, y1, x2, y2], "label": str, "confidence": float, "source": str}, ...]."""

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NullDetector(Detector):
    """Detects nothing."""

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        return []


@dataclass
class ColorClass:
    label: str
    # OpenCV HSV: hue 0-179, saturation and value 0-255. Hue ranges may wrap (red).
    hue_ranges: Sequence[Tuple[int, int]]
    min_sat: int = 120
    min_val: int = 50


# Hues measured from Gazebo-rendered frames of worlds/search_and_rescue.sdf (lighting
# and colour encoding shift them from the SDF diffuse colours): fire 6-8, survivor 15-17.
SEARCH_AND_RESCUE_CLASSES = [
    ColorClass("survivor", [(12, 22)]),                   # orange jacket, diffuse 0.9 0.4 0.1
    ColorClass("fire_hazard", [(0, 10), (172, 179)]),     # red cone, diffuse 0.95 0.15 0.05
    ColorClass("landing_zone", [(100, 120)], min_sat=100),  # blue pad, diffuse 0.1 0.4 0.9
]


class ColorDetector(Detector):
    def __init__(self, classes: Sequence[ColorClass] = SEARCH_AND_RESCUE_CLASSES,
                 min_area_frac: float = 0.0005):
        self.classes = list(classes)
        self.min_area_frac = min_area_frac
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        min_area = self.min_area_frac * bgr.shape[0] * bgr.shape[1]
        out = []
        for cls in self.classes:
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in cls.hue_ranges:
                mask |= cv2.inRange(hsv, (lo, cls.min_sat, cls.min_val), (hi, 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # Fraction of the box covered by the blob: a crude "how blob-like" score.
                fill = float(area) / float(w * h)
                out.append({"bbox": [x, y, x + w, y + h], "label": cls.label,
                            "confidence": round(min(0.95, 0.5 + 0.5 * fill), 3), "source": "color"})
        return out


def make_detector(name: str) -> Detector:
    if name == "none":
        return NullDetector()
    if name == "color":
        return ColorDetector()
    raise ValueError(f"Unknown detector '{name}' (expected 'color' or 'none').")
