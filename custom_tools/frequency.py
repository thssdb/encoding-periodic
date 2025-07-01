import math
import numpy as np
from numpy.typing import NDArray
from typing import Callable


def quantize(x: np.float64, beta: int) -> np.int64:
    return np.int64(math.floor(x / (2**beta) + 0.5))


quantize_vectorized: Callable[[NDArray[np.float64], int], NDArray[np.int64]] = (
    np.vectorize(quantize, otypes=[np.int64])
)


def time_to_frequency(data: NDArray[np.float64]) -> NDArray[np.float64]:
    data_frequency = np.fft.rfft(data)
    data_frequency_real = np.real(data_frequency)
    data_frequency_imag = np.imag(data_frequency)
    return np.concat((data_frequency_real, data_frequency_imag), axis=0)


def time_to_frequency_quantized(
    data: NDArray[np.float64], beta: int
) -> NDArray[np.int64]:
    data_frequency = time_to_frequency(data)
    return quantize_vectorized(data_frequency, beta)


def frequency_to_time(data: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    frequency_length = data.shape[0] // 2
    data_frequency_real = data[:frequency_length]
    data_frequency_imag = data[frequency_length:]
    data_frequency = data_frequency_real + 1j * data_frequency_imag
    data_time = np.fft.irfft(data_frequency, n=length)
    return data_time


def frequency_to_time_quantized(
    data: NDArray[np.int64], length: int, beta: int
) -> NDArray[np.float64]:
    data_frequency = data.astype(np.float64) * (2**beta)
    return frequency_to_time(data_frequency, length)
