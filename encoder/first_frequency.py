from typing import Tuple, List
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from encoder import Encoder
from custom_tools.period import find_period
from algorithm.separate_storage import separate_storage_encode, separate_storage_decode
from algorithm.frequency_encoding import frequency_decode, frequency_encode
from algorithm.separate_storage import get_bit_length_cnt, get_optimal_d


def get_index_all(
    n: int, period: int, count: int, use_trend: bool = True, use_seasonal: bool = True
) -> List[NDArray[np.int64]]:
    n_complete = (n + period - 1) // period * period
    period_count = n_complete // period
    index_trend = np.asarray(list(range(min(count, n_complete // 2))))
    index_seasonal = np.asarray(
        list(
            range(
                0,
                min(count * period_count, n_complete // 2),
                period_count,
            )
        )
    )

    if use_trend and use_seasonal:
        index_trend = np.setdiff1d(index_trend, index_seasonal, assume_unique=True)
        return [index_trend, index_seasonal]
    elif use_trend:
        return [index_trend]
    elif use_seasonal:
        return [index_seasonal]
    else:
        return []


def get_frequency_complete(
    data: NDArray[np.int64], period: int
) -> NDArray[np.complex64]:
    n = data.shape[0]
    data_complete = (
        np.concatenate((data, data[n - period : n // period * period]))
        if n % period != 0
        else data
    )
    frequency_complete = np.fft.fft(data_complete).astype(np.complex64)
    return frequency_complete


def get_freq_and_residual(
    data: NDArray[np.int64],
    period: int,
    count: int,
    beta: int = 0,
    use_trend: bool = True,
    use_seasonal: bool = True,
    use_given_frequency_complete: bool = False,
    given_frequency_complete: NDArray[np.complex64] = np.asarray(
        [], dtype=np.complex64
    ),
) -> Tuple[List[NDArray[np.int64]], NDArray[np.int64]]:
    n = data.shape[0]
    if not use_given_frequency_complete:
        frequency_complete = get_frequency_complete(data, period)
    else:
        frequency_complete = given_frequency_complete

    frequency_complete_indexed = np.zeros_like(frequency_complete, dtype=np.complex64)
    index_all = get_index_all(n, period, count, use_trend, use_seasonal)
    frequency_complete_encoded: List[NDArray[np.int64]] = []

    for index_sequence in index_all:
        frequency_complete_sequence_real = np.asarray(
            np.round(np.real(frequency_complete[index_sequence]) / (2**beta)),
            dtype=np.int64,
        )
        frequency_complete_sequence_imag = np.asarray(
            np.round(np.imag(frequency_complete[index_sequence]) / (2**beta)),
            dtype=np.int64,
        )
        frequency_complete_encoded.append(frequency_complete_sequence_real)
        frequency_complete_encoded.append(frequency_complete_sequence_imag)
        frequency_complete_indexed[
            index_sequence
        ] = frequency_complete_sequence_real.astype(np.complex64) * (
            2**beta
        ) + 1j * frequency_complete_sequence_imag.astype(
            np.complex64
        ) * (
            2**beta
        )
        frequency_complete_indexed[
            -index_sequence
        ] = frequency_complete_sequence_real.astype(np.complex64) * (
            2**beta
        ) - 1j * frequency_complete_sequence_imag.astype(
            np.complex64
        ) * (
            2**beta
        )

    residual: NDArray[np.int64] = (
        data
        - np.round(np.fft.ifft(frequency_complete_indexed).real).astype(np.int64)[:n]
    )

    return frequency_complete_encoded, residual


class FirstFrequencyEncoder(Encoder):
    def __init__(
        self,
        count: int = 10,
        use_beta: bool = True,
        beta: int = 0,
        use_trend: bool = True,
        use_seasonal: bool = True,
    ):
        super().__init__()
        self.count = count
        self.use_beta = use_beta
        self.beta = beta
        self.use_trend = use_trend
        self.use_seasonal = use_seasonal
        self.frequency_part_bit = -1
        self.residual_part_bit = -1
        self.optimal_d = -1
        self.optimal_beta = -1

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        n = len(data)
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return

        period = find_period(data)
        stream.append(BitArray(uint=period, length=MAX_LENGTH.bit_length()))
        if period == 0:
            period = n

        if self.use_beta:
            frequency_complete_encoded, residual = get_freq_and_residual(
                data,
                period,
                self.count,
                self.beta,
                self.use_trend,
                self.use_seasonal,
            )
        else:
            frequency_complete = get_frequency_complete(data, period)
            _, residual = get_freq_and_residual(
                data,
                period,
                self.count,
                0,
                self.use_trend,
                self.use_seasonal,
                use_given_frequency_complete=True,
                given_frequency_complete=frequency_complete,
            )
            optimal_d = get_optimal_d(
                n, get_bit_length_cnt(np.abs(residual).astype(np.uint64))
            )
            optimal_beta = max(0, int(optimal_d + np.log2(n / np.sqrt(2 * self.count))))
            self.optimal_beta = optimal_beta
            stream.append(
                BitArray(
                    uint=optimal_beta, length=MAX_VALUE.bit_length().bit_length() + 2
                )
            )
            frequency_complete_encoded, residual = get_freq_and_residual(
                data,
                period,
                self.count,
                optimal_beta,
                self.use_trend,
                self.use_seasonal,
                use_given_frequency_complete=True,
                given_frequency_complete=frequency_complete,
            )

        start = stream.pos
        for frequency_complete_encoded_part in frequency_complete_encoded:
            frequency_encode(stream, frequency_complete_encoded_part, save_length=False)
        self.frequency_part_bit = int(stream.pos - start)
        start = stream.pos
        self.optimal_d = separate_storage_encode(stream, residual, save_length=False)
        self.residual_part_bit = int(stream.pos - start)

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)

        period = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if period == 0:
            period = n

        n_complete = (n + period - 1) // period * period
        index_all = get_index_all(
            n, period, self.count, self.use_trend, self.use_seasonal
        )

        if self.use_beta:
            beta = self.beta
        else:
            beta = int(stream.read(MAX_VALUE.bit_length().bit_length() + 2).uint)

        frequency_complete_indexed = np.zeros(n_complete, dtype=np.complex64)
        for index_sequence in index_all:
            frequency_complete_sequence_real = frequency_decode(
                stream, save_length=False, given_length=len(index_sequence)
            )
            frequency_complete_sequence_imag = frequency_decode(
                stream, save_length=False, given_length=len(index_sequence)
            )
            frequency_complete_indexed[
                index_sequence
            ] = frequency_complete_sequence_real.astype(np.complex64) * (
                2**beta
            ) + 1j * frequency_complete_sequence_imag.astype(
                np.complex64
            ) * (
                2**beta
            )
            frequency_complete_indexed[
                -index_sequence
            ] = frequency_complete_sequence_real.astype(np.complex64) * (
                2**beta
            ) - 1j * frequency_complete_sequence_imag.astype(
                np.complex64
            ) * (
                2**beta
            )

        residual = separate_storage_decode(stream, save_length=False, given_length=n)
        data = (
            np.round(np.fft.ifft(frequency_complete_indexed).real).astype(np.int64)[:n]
            + residual
        )

        return data

    def get_name(self) -> str:
        base_name = "First-Frequency"
        if not self.use_trend:
            base_name += " (No Trend)"
        if not self.use_seasonal:
            base_name += " (No Seasonal)"
        return base_name

    def exp(self, data: NDArray[np.int64]) -> pd.DataFrame:
        result = super().exp(data)
        result["frequency_part"] = self.frequency_part_bit // 8
        result["residual_part"] = self.residual_part_bit // 8
        result["count"] = self.count
        result["beta"] = self.beta
        result["optimal_d"] = self.optimal_d
        result["optimal_beta"] = self.optimal_beta
        return result
