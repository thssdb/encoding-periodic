import numpy as np
from numpy.typing import NDArray
from custom_tools.run_length_encode import run_length_encode


def test_run_length_encode():
    data = np.array([1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 1])
    expected_output = (
        np.asarray([1, 2, 3, 1], dtype=np.int64),
        np.array([2, 3, 2, 4], dtype=np.int64),
    )

    encoded = run_length_encode(data)

    assert np.array_equal(encoded[0], expected_output[0]), "Values do not match"
    assert np.array_equal(encoded[1], expected_output[1]), "Counts do not match"
