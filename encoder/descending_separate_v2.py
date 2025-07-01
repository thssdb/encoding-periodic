import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.separate_storage import separate_storage_encode, separate_storage_decode
from algorithm.frequency_encoding import frequency_decode, frequency_encode


class DescendingSeparateEncoder(Encoder):
    def frequency_estimate(
        self, frequency: NDArray[np.int64], beta_min: int, beta_max: int
    ) -> NDArray[np.int64]:
        index_bit_length = int(max(frequency.shape[0] - 1, 1)).bit_length()

        frequency_bit_length = np.ceil(np.log2(np.abs(frequency) + 1)).astype(np.int64)
        frequency_bit_length_count = np.bincount(frequency_bit_length)

        result = np.zeros(beta_max - beta_min + 1, dtype=np.int64)

        for i, count in enumerate(frequency_bit_length_count):
            for beta in range(beta_min, beta_max + 1):
                if i <= beta:
                    continue
                result[beta - beta_min] += count * (index_bit_length + (i - beta + 1))

        return result

    def residual_estimate(
        self,
        n: int,
        frequency_real: NDArray[np.int64],
        frequency_imag: NDArray[np.int64],
        beta_min: int,
        beta_max: int,
    ) -> NDArray[np.int64]:
        square_sum_diff = np.zeros(beta_max - beta_min + 1, dtype=np.float64)
        partial_count_diff = np.zeros(beta_max - beta_min + 1, dtype=np.int64)

        for i in range(n):
            real: np.int64 = (
                frequency_real[i]
                if i < frequency_real.shape[0]
                else frequency_real[n - i]
            )
            real_bit_length = int(np.ceil(np.log2(np.abs(real) + 1)))
            if real_bit_length - 1 >= beta_min:
                partial_count_diff[0] += 1
                if real_bit_length <= beta_max:
                    partial_count_diff[real_bit_length - beta_min] -= 1
            if real_bit_length >= beta_min and real_bit_length <= beta_max:
                square_sum_diff[real_bit_length - beta_min] += (
                    real.astype(np.float64) ** 2
                )

            imag: np.int64 = (
                frequency_imag[i]
                if i < frequency_imag.shape[0]
                else frequency_imag[n - i]
            )
            imag_bit_length = int(np.ceil(np.log2(np.abs(imag) + 1)))
            if imag_bit_length - 1 >= beta_min:
                partial_count_diff[0] += 1
                if imag_bit_length <= beta_max:
                    partial_count_diff[imag_bit_length - beta_min] -= 1
            if imag_bit_length >= beta_min and imag_bit_length <= beta_max:
                square_sum_diff[imag_bit_length - beta_min] += (
                    imag.astype(np.float64) ** 2
                )

        square_sum = np.cumsum(square_sum_diff)
        partial_count = np.cumsum(partial_count_diff)

        result = np.zeros(beta_max - beta_min + 1, dtype=np.int64)

        for beta in range(beta_min, beta_max + 1):
            square_sum_beta: np.float64 = (
                square_sum[beta - beta_min]
                + partial_count[beta - beta_min] * (2**beta) ** 2 // 3
            )
            optimal_d: np.int64 = np.ceil(
                np.log2(np.sqrt(square_sum_beta) / n + 1)
            ).astype(np.int64)
            result[beta - beta_min] = (optimal_d + 2) * n

        return result

    def __init__(self, min_beta: int = 5, max_beta: int = 20):
        super().__init__()
        self.min_beta = min_beta
        self.max_beta = max_beta

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        n = data.shape[0]
        stream.append(BitStream(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return

        frequency_value = np.fft.rfft(data)
        frequency_value_real = np.round(np.real(frequency_value)).astype(np.int64)
        frequency_value_imag = np.round(np.imag(frequency_value)).astype(np.int64)

        estimate_encoding_length = (
            self.frequency_estimate(frequency_value_real, self.min_beta, self.max_beta)
            + self.frequency_estimate(
                frequency_value_imag, self.min_beta, self.max_beta
            )
            + self.residual_estimate(
                n,
                frequency_value_real,
                frequency_value_imag,
                self.min_beta,
                self.max_beta,
            )
        )

        optimal_beta = np.argmin(estimate_encoding_length) + self.min_beta
        stream.append(
            BitArray(uint=optimal_beta, length=MAX_VALUE.bit_length().bit_length() + 2)
        )

        frequency_value_quantized_real = np.round(
            frequency_value_real / (2**optimal_beta)
        ).astype(np.int64)
        frequency_value_quantized_imag = np.round(
            frequency_value_imag / (2**optimal_beta)
        ).astype(np.int64)
        frequency_value_quantized = frequency_value_quantized_real.astype(
            np.complex128
        ) * (2**optimal_beta) + 1j * frequency_value_quantized_imag.astype(
            np.complex128
        ) * (
            2**optimal_beta
        )
        frequency_encode(stream, frequency_value_quantized_real, save_length=False)
        frequency_encode(stream, frequency_value_quantized_imag, save_length=False)

        residual = data - np.round(np.fft.irfft(frequency_value_quantized, n=n)).astype(
            np.int64
        )
        separate_storage_encode(stream, residual, save_length=False)

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)

        optimal_beta = int(stream.read(MAX_VALUE.bit_length().bit_length() + 2).uint)

        frequency_value_quantized_real = frequency_decode(
            stream, save_length=False, given_length=(n // 2 + 1)
        )
        frequency_value_quantized_imag = frequency_decode(
            stream, save_length=False, given_length=(n // 2 + 1)
        )
        frequency_value_quantized = frequency_value_quantized_real.astype(
            np.complex128
        ) * (2**optimal_beta) + 1j * frequency_value_quantized_imag.astype(
            np.complex128
        ) * (
            2**optimal_beta
        )

        residual = separate_storage_decode(stream, save_length=False, given_length=n)
        data = (
            np.round(np.fft.irfft(frequency_value_quantized, n=n)).astype(np.int64)
            + residual
        )
        return data

    def get_name(self) -> str:
        return "Descending-Separate-V2"
