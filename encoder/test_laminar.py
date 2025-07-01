import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from encoder import Encoder
from encoder.laminar import LaminarEncoder


def test_laminar_encoder():
    data = np.array(
        [100, -50, 9, -82, 3, -8, 7, 10, 10, -5, 0, 2, 0, 1, 0, 0, 0], dtype=np.int64
    )
    encoder = LaminarEncoder()
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "Laminar"


def test_laminar_encoder_sparse_mode():
    data = np.array(
        [100, -50, 0, 0, 0, -8, 7, 0, 10, 0, 0, 2, 0, 1, 0, 0, 0], dtype=np.int64
    )
    encoder = LaminarEncoder(sparse_mode=True)
    original_data = data.copy()
    result = encoder.exp(data)

    assert np.array_equal(data, original_data)
    assert (result["encoder"].to_numpy())[0] == "Laminar"


def test_laminar_encoder_sparse_mode_zero():
    data = np.array([0, 0, 0], dtype=np.int64)
    encoder = LaminarEncoder(sparse_mode=True)
    original_data = data.copy()
    result = encoder.exp(data)

    assert np.array_equal(data, original_data)
    assert (result["encoder"].to_numpy())[0] == "Laminar"
