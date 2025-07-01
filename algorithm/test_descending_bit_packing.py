import numpy as np
from bitstring import BitStream
from algorithm.descending_bit_packing import (
    descending_bit_packing_encode,
    descending_bit_packing_decode,
)


def test_descending_bit_packing():
    data = np.array([100, 0, 9, 82, 0, 0, 7, 0, 0, 0], dtype=np.uint64)
    stream = BitStream()
    descending_bit_packing_encode(stream, data)
    stream.pos = 0
    assert np.array_equal(data, descending_bit_packing_decode(stream))


def test_descending_bit_packing_given_length():
    data = np.array([100, 0, 9, 82, 0, 0, 7, 0, 0, 0], dtype=np.uint64)
    given_length = data.shape[0]
    stream = BitStream()
    descending_bit_packing_encode(stream, data, save_length=False)
    stream.pos = 0
    assert np.array_equal(
        data,
        descending_bit_packing_decode(
            stream, save_length=False, given_length=given_length
        ),
    )
