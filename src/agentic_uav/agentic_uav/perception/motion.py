"""Yaw rate from PX4 heading samples, measured over a time window.

Differencing consecutive samples (~10 ms apart) turns milliradians of estimator
noise into apparent turn rates of 0.1-0.5 rad/s, so the rate is taken between the
newest sample and one at least `window_s` older.
"""

import math
from collections import deque


class YawRateEstimator:
    def __init__(self, window_s: float = 0.3, max_age_s: float = 2.0):
        self.window_us = int(window_s * 1e6)
        self.max_age_us = int(max_age_s * 1e6)
        self._samples = deque()  # (timestamp_us, heading_rad)

    def add(self, timestamp_us: int, heading: float):
        if not math.isfinite(heading) or (self._samples and timestamp_us <= self._samples[-1][0]):
            return
        self._samples.append((timestamp_us, heading))
        while self._samples and timestamp_us - self._samples[0][0] > self.max_age_us:
            self._samples.popleft()

    def rate(self) -> float:
        """rad/s over the newest window; 0 until a full window of samples exists."""
        if not self._samples:
            return 0.0
        t1, h1 = self._samples[-1]
        older = [(t, h) for t, h in self._samples if t1 - t >= self.window_us]
        if not older:
            return 0.0
        t0, h0 = older[-1]
        return math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)) / ((t1 - t0) * 1e-6)
