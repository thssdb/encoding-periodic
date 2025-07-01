import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)
from encoder import Encoder
from encoder.period_diff_separate import PeriodDiffSeparateEncoder


def test_period_diff_separate_encoder():
    np.random.seed(0)
    noise = np.random.normal(0, 0.1, 100)
    signal_noisy = ((np.sin(np.linspace(0, 20 * np.pi, 100)) + noise) * 100).astype(
        np.int64
    )
    encoder = PeriodDiffSeparateEncoder()
    result = encoder.exp(signal_noisy)
    assert (result["encoder"].to_numpy())[0] == "Period-Diff-Separate"
