import numpy as np
from bitstring import BitStream
from algorithm.bit_packing import bit_packing_encode, bit_packing_decode


def test_bit_packing():
    data = np.array([100, 0, 9, 82, 0, 0, 7, 0, 0, 0], dtype=np.uint64)
    bit_length = 8
    stream = BitStream()

    bit_packing_encode(stream, data, bit_length)

    stream.pos = 0

    decoded_data = bit_packing_decode(stream)

    assert np.array_equal(data, decoded_data)


def test_bit_packing_given_length():
    data = np.array([100, 0, 9, 82, 0, 0, 7, 0, 0, 0], dtype=np.uint64)
    bit_length = 8
    given_length = data.shape[0]
    stream = BitStream()

    bit_packing_encode(stream, data, bit_length, save_length=False)

    stream.pos = 0

    decoded_data = bit_packing_decode(
        stream, save_length=False, given_length=given_length
    )

    assert np.array_equal(data, decoded_data)
