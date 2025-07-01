import numpy as np
from numpy.typing import NDArray
from custom_tools.frequency import (
    time_to_frequency,
    frequency_to_time,
    time_to_frequency_quantized,
    frequency_to_time_quantized,
)


def test_time_frequency():
    np.random.seed(42)
    length = 1000
    data = np.random.rand(length).astype(np.float64)

    data_frequency = time_to_frequency(data)

    data_time = frequency_to_time(data_frequency, length)

    assert np.allclose(
        data, data_time, atol=1e-6
    ), "The reconstructed data does not match the original data."


def test_time_frequency_quantized():
    np.random.seed(42)
    length = 1000
    data = np.random.rand(length).astype(np.float64)

    beta = 1
    data_frequency_quantized = time_to_frequency_quantized(data, beta)
    data_time_quantized = frequency_to_time_quantized(
        data_frequency_quantized, length, beta
    )
    assert np.allclose(
        data, data_time_quantized, atol=0.5
    ), "The reconstructed data does not match the original data after quantization."
