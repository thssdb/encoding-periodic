import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from encoder import Encoder
from encoder.sprintz import SprintzEncoder


def test_diff_separate_encoder():
    data = np.array([100, -50, 9, -82, 3, -8, 7, 10, 10, -5] * 1000, dtype=np.int64)
    encoder = SprintzEncoder()
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "Sprintz"
