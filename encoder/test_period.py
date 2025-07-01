import numpy as np
from encoder.period import PeriodEncoder


def test_diff_separate_encoder():
    np.random.seed(0)
    length = 100
    noise = np.random.normal(0, 0.1, length)
    signal_noisy = ((np.sin(np.linspace(0, 20 * np.pi, length)) + noise) * 100).astype(
        np.int64
    )
    encoder = PeriodEncoder()
    result = encoder.exp(signal_noisy)
    assert (result["encoder"].to_numpy())[0] == "Period"


def test_diff_separate_encoder_odd_length():
    np.random.seed(0)
    length = 1001
    noise = np.random.normal(0, 0.1, length)
    signal_noisy = ((np.sin(np.linspace(0, 20 * np.pi, length)) + noise) * 100).astype(
        np.int64
    )
    encoder = PeriodEncoder()
    result = encoder.exp(signal_noisy)
    assert (result["encoder"].to_numpy())[0] == "Period"
