import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from encoder import Encoder
from encoder.rle import RLEEncoder


def test_rle_encoder():
    data = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4] * 1000, dtype=np.int64)
    encoder = RLEEncoder()
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "RLE"


def test_rle_encoder_empty():
    data = np.array([], dtype=np.int64)
    encoder = RLEEncoder()
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "RLE"


def test_rle_encoder_special_mode():
    data = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4], dtype=np.int64)
    encoder = RLEEncoder(
        special_mode=True, special_mode_value=2, special_mode_max_length=1
    )
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "RLE"
