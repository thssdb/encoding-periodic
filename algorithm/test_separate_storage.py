import numpy as np
from bitstring import BitStream
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)


def test_separate_storage():
    data = np.array([100, -50, 9, -82, 3, -8, 7, 10, 9, -5], dtype=np.int64)
    stream = BitStream()
    separate_storage_encode(stream, data)
    stream.pos = 0
    decoded_data = separate_storage_decode(stream)
    assert np.array_equal(data, decoded_data)


def test_separate_storage_zero_d():
    data = np.array([100, -50, 9, -82, 0, 0, 7, 0, 0, 0], dtype=np.int64)
    stream = BitStream()
    separate_storage_encode(stream, data)
    stream.pos = 0
    decoded_data = separate_storage_decode(stream)
    assert np.array_equal(data, decoded_data)


def test_separate_storage_given_length():
    data = np.array([100, -50, 9, -82, 3, -8, 7, 10, 9, -5], dtype=np.int64)
    stream = BitStream()
    separate_storage_encode(stream, data, save_length=False)
    stream.pos = 0
    decoded_data = separate_storage_decode(
        stream, save_length=False, given_length=data.shape[0]
    )
    assert np.array_equal(data, decoded_data)
