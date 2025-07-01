import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.separate_storage import separate_storage_encode, separate_storage_decode
from encoder.laminar_hybrid import LaminarHybridEncoder
from typing import Tuple


class FLEAEncoder(Encoder):
    def frequency_estimate(self, frequency: NDArray[np.int64]) -> NDArray[np.int64]:
        bit_width = np.ceil(np.log2(np.abs(frequency) + 1)).astype(np.int64)
        bit_width_suffix_max = np.flip(
            np.maximum.accumulate(np.flip(bit_width))
        ).astype(np.int64)
        return self.laminar_hybrid_encoder.estimate_length(
            frequency, bit_width, bit_width_suffix_max
        ) + self.laminar_hybrid_encoder.estimate_value(
            frequency, bit_width, bit_width_suffix_max
        )

    def residual_estimate(
        self,
        n: int,
        frequency_real: NDArray[np.int64],
        frequency_imag: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        square_sum_diff = np.zeros(self.max_beta - self.min_beta + 1, dtype=np.float64)
        partial_count_diff = np.zeros(self.max_beta - self.min_beta + 1, dtype=np.int64)

        for i in range(n):
            real: np.int64 = (
                frequency_real[i]
                if i < frequency_real.shape[0]
                else frequency_real[n - i]
            )
            real_bit_length = int(np.ceil(np.log2(np.abs(real) + 1)))
            if real_bit_length - 1 >= self.min_beta:
                partial_count_diff[0] += 1
                if real_bit_length <= self.max_beta:
                    partial_count_diff[real_bit_length - self.min_beta] -= 1
            if real_bit_length >= self.min_beta and real_bit_length <= self.max_beta:
                square_sum_diff[real_bit_length - self.min_beta] += (
                    real.astype(np.float64) ** 2
                )

            imag: np.int64 = (
                frequency_imag[i]
                if i < frequency_imag.shape[0]
                else frequency_imag[n - i]
            )
            imag_bit_length = int(np.ceil(np.log2(np.abs(imag) + 1)))
            if imag_bit_length - 1 >= self.min_beta:
                partial_count_diff[0] += 1
                if imag_bit_length <= self.max_beta:
                    partial_count_diff[imag_bit_length - self.min_beta] -= 1
            if imag_bit_length >= self.min_beta and imag_bit_length <= self.max_beta:
                square_sum_diff[imag_bit_length - self.min_beta] += (
                    imag.astype(np.float64) ** 2
                )

        square_sum = np.cumsum(square_sum_diff)
        partial_count = np.cumsum(partial_count_diff)

        result = np.zeros(self.max_beta - self.min_beta + 1, dtype=np.int64)

        for beta in range(self.min_beta, self.max_beta + 1):
            square_sum_beta: np.float64 = (
                square_sum[beta - self.min_beta]
                + partial_count[beta - self.min_beta] * (2**beta) ** 2 // 3
            )
            optimal_d: np.int64 = np.ceil(
                np.log2(np.sqrt(square_sum_beta) / n + 1)
            ).astype(np.int64)
            result[beta - self.min_beta] = (optimal_d + 2) * n

        return result

    def init_laminar_hybrid_encoder(self) -> LaminarHybridEncoder:
        return LaminarHybridEncoder(
            min_beta=self.min_beta,
            max_beta=self.max_beta,
        )

    def __init__(
        self,
        min_beta: int = 5,
        max_beta: int = 20,
    ):
        super().__init__()
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.laminar_hybrid_encoder = self.init_laminar_hybrid_encoder()

    def get_optimal_beta(
        self,
        n: int,
        frequency_value_real: NDArray[np.int64],
        frequency_value_imag: NDArray[np.int64],
    ):
        estimate_encoding_length = (
            self.frequency_estimate(frequency_value_real)
            + self.frequency_estimate(frequency_value_imag)
            + self.residual_estimate(
                n,
                frequency_value_real,
                frequency_value_imag,
            )
        )
        optimal_beta = np.argmin(estimate_encoding_length) + self.min_beta
        return optimal_beta

    def encode_frequency(
        self,
        stream: BitStream,
        frequency_value_quantized_real: NDArray[np.int64],
        frequency_value_quantized_imag: NDArray[np.int64],
    ):
        self.laminar_hybrid_encoder.encode_stream_param(
            stream, frequency_value_quantized_real, save_length=False
        )
        self.laminar_hybrid_encoder.encode_stream_param(
            stream, frequency_value_quantized_imag, save_length=False
        )

    def encode_residual(
        self,
        stream: BitStream,
        residual: NDArray[np.int64],
    ):
        separate_storage_encode(stream, residual, save_length=False)

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        n = data.shape[0]
        stream.append(BitStream(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return

        frequency_value = np.fft.rfft(data)
        frequency_value_real = np.round(np.real(frequency_value)).astype(np.int64)
        frequency_value_imag = np.round(np.imag(frequency_value)).astype(np.int64)

        optimal_beta = self.get_optimal_beta(
            n,
            frequency_value_real,
            frequency_value_imag,
        )

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
        self.encode_frequency(
            stream,
            frequency_value_quantized_real,
            frequency_value_quantized_imag,
        )

        residual = data - np.round(np.fft.irfft(frequency_value_quantized, n=n)).astype(
            np.int64
        )
        self.encode_residual(stream, residual)

    def decode_frequency(
        self, stream: BitStream, n: int
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        frequency_value_quantized_real = self.laminar_hybrid_encoder.decode_param(
            stream, save_length=False, given_length=n // 2 + 1
        )
        frequency_value_quantized_imag = self.laminar_hybrid_encoder.decode_param(
            stream, save_length=False, given_length=n // 2 + 1
        )
        return (
            frequency_value_quantized_real.astype(np.int64),
            frequency_value_quantized_imag.astype(np.int64),
        )

    def decode_residual(self, stream: BitStream, n: int) -> NDArray[np.int64]:
        return separate_storage_decode(stream, save_length=False, given_length=n)

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)

        optimal_beta = int(stream.read(MAX_VALUE.bit_length().bit_length() + 2).uint)

        frequency_value_quantized_real, frequency_value_quantized_imag = (
            self.decode_frequency(stream, n)
        )
        frequency_value_quantized = frequency_value_quantized_real.astype(
            np.complex128
        ) * (2**optimal_beta) + 1j * frequency_value_quantized_imag.astype(
            np.complex128
        ) * (
            2**optimal_beta
        )

        residual = self.decode_residual(stream, n)
        data = (
            np.round(np.fft.irfft(frequency_value_quantized, n=n)).astype(np.int64)
            + residual
        )
        return data

    def get_name(self) -> str:
        return "FLEA"
