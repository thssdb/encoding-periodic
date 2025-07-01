import numpy as np
from bitstring import BitStream
from algorithm.bit_packing_signed_grouped import (
    bit_packing_signed_grouped_encode,
    bit_packing_signed_grouped_decode,
)


def test_bit_packing_signed_grouped():
    data = np.array([-100, 0, 9, 82, 0, 0, -7, 0, 0, 0] * 1000, dtype=np.int64)
    stream = BitStream()

    bit_packing_signed_grouped_encode(stream, data)

    stream.pos = 0

    decoded_data = bit_packing_signed_grouped_decode(stream)

    assert np.array_equal(data, decoded_data)
