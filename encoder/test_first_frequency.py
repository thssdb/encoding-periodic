import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)
from encoder import Encoder
from encoder.first_frequency import FirstFrequencyEncoder


def get_random_signal(length: int) -> NDArray[np.int64]:
    np.random.seed(0)
    noise = np.random.normal(0, 0.1, length)
    signal = ((np.sin(np.linspace(0, 20 * np.pi, length)) + noise) * 100).astype(
        np.int64
    )
    return signal


def test_first_frequency_encoder():
    signal = get_random_signal(100)
    encoder = FirstFrequencyEncoder()
    result = encoder.exp(signal)
    assert (result["encoder"].to_numpy())[0] == "First-Frequency"


def test_first_frequency_encoder_not_use_trend():
    signal = get_random_signal(100)
    encoder = FirstFrequencyEncoder(use_trend=False)
    result = encoder.exp(signal)
    assert (result["encoder"].to_numpy())[0] == "First-Frequency (No Trend)"


def test_first_frequency_encoder_not_use_seasonal():
    signal = get_random_signal(100)
    encoder = FirstFrequencyEncoder(use_seasonal=False)
    result = encoder.exp(signal)
    assert (result["encoder"].to_numpy())[0] == "First-Frequency (No Seasonal)"


def test_first_frequency_encoder_beta():
    signal = get_random_signal(100)
    encoder = FirstFrequencyEncoder(use_beta=False)
    result = encoder.exp(signal)
    assert (result["encoder"].to_numpy())[0] == "First-Frequency"
