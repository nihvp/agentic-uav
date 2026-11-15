import math
import random

from agentic_uav.perception.motion import YawRateEstimator


def feed(est, rate, seconds, noise=0.0, hz=100, start=0.0):
    rng = random.Random(1)
    for i in range(int(seconds * hz)):
        t = i / hz
        est.add(int(t * 1e6), math.atan2(math.sin(start + rate * t), math.cos(start + rate * t)) + rng.gauss(0, noise))


def test_hover_noise_is_not_a_turn():
    est = YawRateEstimator()
    feed(est, 0.0, 1.0, noise=0.003)      # 3 mrad jitter at 100 Hz: ~0.4 rad/s if differenced per sample
    assert abs(est.rate()) < 0.05


def test_real_turn_is_measured_across_the_wrap():
    est = YawRateEstimator()
    feed(est, 0.5, 1.0, start=math.pi - 0.2)  # crosses +pi -> -pi
    assert abs(est.rate() - 0.5) < 0.01


def test_no_rate_until_window_filled():
    est = YawRateEstimator()
    feed(est, 1.0, 0.1)
    assert est.rate() == 0.0
