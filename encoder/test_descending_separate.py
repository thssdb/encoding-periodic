import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from encoder import Encoder
from encoder.descending_separate import DescendingSeparateEncoder


def test_descending_separate_encoder():
    data = np.array([100, -50, 9, -82, 3, -8, 7, 10, 9, -5] * 1000, dtype=np.int64)
    encoder = DescendingSeparateEncoder(beta=5)
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "Descending-Separate"


def test_descending_separate_encoder_not_with_descending():
    data = np.array([100, -50, 9, -82, 3, -8, 7, 10, 9, -5] * 1000, dtype=np.int64)
    encoder = DescendingSeparateEncoder(beta=5, use_descending=False)
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "Descending-Separate (Grouped)"
