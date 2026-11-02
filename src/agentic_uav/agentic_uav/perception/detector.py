"""Detector interface. Phase 2 adds a YOLO implementation."""

from typing import Any, Dict, List

import numpy as np


class Detector:
    """Returns [{"bbox": [x1, y1, x2, y2], "label": str, "confidence": float}, ...]."""

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NullDetector(Detector):
    """Detects nothing. Keeps the pipeline running honestly until Phase 2."""

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        return []


def make_detector(name: str) -> Detector:
    if name == "none":
        return NullDetector()
    raise ValueError(f"Unknown detector '{name}' (Phase 2 adds 'yolo').")
