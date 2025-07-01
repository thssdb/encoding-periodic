import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from algorithm.constants import MAX_LENGTH
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)
from encoder import Encoder
from encoder.diff_separate import DiffSeparateEncoder
from custom_tools.period import find_period


class PeriodDiffSeparateEncoder(DiffSeparateEncoder):
    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        period = find_period(data)
        stream.append(BitStream(uint=period, length=MAX_LENGTH.bit_length()))
        if period == 0:
            super().encode_stream(stream, data)
            return
        n = data.shape[0]
        stream.append(BitStream(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return
        residual = np.zeros(n, dtype=np.int64)
        residual[0] = data[0]
        for i in range(1, min(period + 1, n)):
            residual[i] = data[i] - data[i - 1]
        for i in range(min(period + 1, n), n):
            residual[i] = data[i] - (
                data[i - 1] + data[i - period] - data[i - period - 1]
            )
        separate_storage_encode(stream, residual, save_length=False)
        return

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        period = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if period == 0:
            return super().decode(stream)
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)
        residual = separate_storage_decode(stream, save_length=False, given_length=n)
        data = np.zeros(n, dtype=np.int64)
        data[0] = residual[0]
        for i in range(1, min(period + 1, n)):
            data[i] = data[i - 1] + residual[i]
        for i in range(min(period + 1, n), n):
            data[i] = (
                data[i - 1] + data[i - period] - data[i - period - 1] + residual[i]
            )
        return data

    def get_name(self) -> str:
        return "Period-Diff-Separate"
