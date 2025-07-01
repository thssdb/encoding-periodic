import numpy as np
import random
from bitstring import BitStream
from algorithm.frequency_encoding import frequency_encode, frequency_decode


def test_frequency_encoding():
    data = np.array(
        [100]
        + [random.randint(-1, 1) for _ in range(100)]
        + [0] * 100
        + [1] * 10
        + [0] * 100,
        dtype=np.int64,
    )
    stream = BitStream()

    frequency_encode(stream, data)

    stream.pos = 0

    decoded_data = frequency_decode(stream)

    assert np.array_equal(data, decoded_data)
