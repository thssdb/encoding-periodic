import numpy as np
from custom_tools.period import find_period


def test_find_period():
    signal1 = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    period1 = find_period(signal1)
    assert period1 == 4, f"Expected period 4, got {period1}"


def test_find_period_with_constant_signal():
    signal_constant = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    period_constant = find_period(signal_constant)
    assert period_constant == 0, f"Expected no period, got {period_constant}"


def test_find_period_with_near_constant_signal():
    signal_near_constant = np.array([2.0, 2.00000001, 2.0, 1.99999999, 2.0])
    period_near_constant = find_period(signal_near_constant)
    assert period_near_constant == 0, f"Expected no period, got {period_near_constant}"


def test_find_period_with_zero_signal():
    signal_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    period_zero = find_period(signal_zero)
    assert period_zero == 0, f"Expected no period, got {period_zero}"


def test_find_period_with_noise():
    np.random.seed(0)
    noise = np.random.normal(0, 0.1, 100)
    signal_noisy = ((np.sin(np.linspace(0, 20 * np.pi, 100)) + noise) * 100).astype(
        np.int64
    )
    period_noisy = find_period(signal_noisy)
    assert period_noisy > 0, f"Expected a positive period, got {period_noisy}"
