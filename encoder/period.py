import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple
from bitstring import BitStream
from encoder.diff_separate import DiffSeparateEncoder
from custom_tools.period import find_period
from custom_tools.frequency import (
    time_to_frequency_quantized,
    frequency_to_time_quantized,
)
from algorithm.constants import MAX_LENGTH
from algorithm.descending_bit_packing_signed import (
    descending_bit_packing_signed_decode,
    descending_bit_packing_signed_encode,
)


def get_freq_and_residual(
    data: NDArray[np.int64], period: int, beta: int
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = data.shape[0]

    period_part = np.zeros(period, dtype=np.float64)
    for i in range(0, n, period):
        if i + period <= n:
            period_part += data[i : i + period]
        else:
            period_part += np.concatenate((data[i:], data[n - period : i]))
    period_part /= (n + period - 1) // period

    period_part_freq = time_to_frequency_quantized(period_part, beta)
    period_part_lossy = np.round(
        frequency_to_time_quantized(period_part_freq, period, beta)
    ).astype(np.int64)

    residual: NDArray[np.int64] = (
        data - np.tile(period_part_lossy, (n + period - 1) // period)[:n]
    )

    return period_part_freq, residual


class PeriodEncoder(DiffSeparateEncoder):
    def __init__(self, beta: int = 0):
        super().__init__()
        self.beta = beta
        self.period = -1
        self.frequency_part_bit = -1
        self.residual_part_bit = -1

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        period = find_period(data)
        self.period = period
        stream.append(BitStream(uint=period, length=MAX_LENGTH.bit_length()))

        if period == 0:
            self.frequency_part_bit = 0
            start = stream.pos
            super().encode_stream(stream, data)
            self.residual_part_bit = int(stream.pos - start)
            return

        period_part_freq, residual = get_freq_and_residual(data, period, self.beta)

        start = stream.pos
        super().encode_stream(stream, residual)
        self.residual_part_bit = int(stream.pos - start)

        start = stream.pos
        descending_bit_packing_signed_encode(
            stream, period_part_freq, save_length=False
        )
        self.frequency_part_bit = int(stream.pos - start)

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        period = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if period == 0:
            return super().decode(stream)

        residual = super().decode(stream)
        n = residual.shape[0]

        period_part_freq = descending_bit_packing_signed_decode(
            stream, save_length=False, given_length=(period // 2 + 1) * 2
        )
        period_part_lossy = np.round(
            frequency_to_time_quantized(period_part_freq, period, self.beta)
        ).astype(np.int64)

        data = residual + np.tile(period_part_lossy, (n + period - 1) // period)[:n]
        return data

    def get_name(self) -> str:
        return "Period"

    def exp(self, data: NDArray[np.int64]) -> pd.DataFrame:
        result = super().exp(data)
        result["frequency_part"] = self.frequency_part_bit // 8
        result["residual_part"] = self.residual_part_bit // 8
        result["beta"] = self.beta
        result["period"] = self.period
        return result
