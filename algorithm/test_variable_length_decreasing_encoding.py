import numpy as np
from bitstring import BitStream
from algorithm.variable_length_decreasing_encoding import (
    variable_length_decreasing_encode,
    variable_length_decreasing_decode,
)


def test_variable_length_decreasing_encoding():
    data = np.array([-100, 0, 9, -82, 0, 0, 7, 0, 0, 0], dtype=np.int64)
    stream = BitStream()

    variable_length_decreasing_encode(stream, data)

    stream.pos = 0

    decoded_data = variable_length_decreasing_decode(stream)

    assert np.array_equal(data, decoded_data)
