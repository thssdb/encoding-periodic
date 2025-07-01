import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)
from encoder import Encoder
from encoder.diff_separate import DiffSeparateEncoder


def test_diff_separate_encoder():
    data = np.array([100, -50, 9, -82, 3, -8, 7, 10, 9, -5], dtype=np.int64)
    encoder = DiffSeparateEncoder()
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "Diff-Separate"
