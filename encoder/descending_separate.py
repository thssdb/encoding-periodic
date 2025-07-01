from bitstring import BitStream
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from encoder import Encoder
from algorithm.constants import MAX_VALUE, MAX_LENGTH
from algorithm.descending_bit_packing_signed import (
    descending_bit_packing_signed_decode,
    descending_bit_packing_signed_encode,
)
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)
from algorithm.frequency_encoding import (
    frequency_decode,
    frequency_encode,
)


class DescendingSeparateEncoder(Encoder):
    def __init__(self, beta: int, use_descending: bool = True):
        super().__init__()
        self.beta = np.int64(beta)
        self.use_descending = use_descending
        self.frequency_encoding_length = -1
        self.residual_encoding_length = -1

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        n = data.shape[0]
        stream.append(BitStream(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return

        frequency_value = np.fft.rfft(data)
        frequency_value_quantized_real = np.round(
            np.real(frequency_value) / (2**self.beta)
        ).astype(np.int64)
        frequency_value_quantized_imag = np.round(
            np.imag(frequency_value) / (2**self.beta)
        ).astype(np.int64)

        frequency_value_quantized = frequency_value_quantized_real.astype(
            np.complex64
        ) * (2**self.beta) + 1j * frequency_value_quantized_imag.astype(
            np.complex64
        ) * (
            2**self.beta
        )

        residual = data - np.round(np.fft.irfft(frequency_value_quantized, n=n)).astype(
            np.int64
        )

        start = stream.pos
        if self.use_descending:
            descending_bit_packing_signed_encode(
                stream, frequency_value_quantized_real, save_length=False
            )
            descending_bit_packing_signed_encode(
                stream, frequency_value_quantized_imag, save_length=False
            )
        else:
            frequency_encode(stream, frequency_value_quantized_real, save_length=False)
            frequency_encode(stream, frequency_value_quantized_imag, save_length=False)
        self.frequency_encoding_length = stream.pos - start
        start = stream.pos
        separate_storage_encode(stream, residual, save_length=False)
        self.residual_encoding_length = stream.pos - start

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)

        if self.use_descending:
            frequency_value_quantized_real = descending_bit_packing_signed_decode(
                stream, save_length=False, given_length=(n // 2 + 1)
            )
            frequency_value_quantized_imag = descending_bit_packing_signed_decode(
                stream, save_length=False, given_length=(n // 2 + 1)
            )
        else:
            frequency_value_quantized_real = frequency_decode(
                stream, save_length=False, given_length=(n // 2 + 1)
            )
            frequency_value_quantized_imag = frequency_decode(
                stream, save_length=False, given_length=(n // 2 + 1)
            )

        frequency_value_quantized = frequency_value_quantized_real.astype(
            np.complex64
        ) * (2**self.beta) + 1j * frequency_value_quantized_imag.astype(
            np.complex64
        ) * (
            2**self.beta
        )

        residual = separate_storage_decode(stream, save_length=False, given_length=n)

        data = (
            np.round(np.fft.irfft(frequency_value_quantized, n=n)).astype(np.int64)
            + residual
        )
        return data

    def get_name(self) -> str:
        if self.use_descending:
            return "Descending-Separate"
        else:
            return "Descending-Separate (Grouped)"

    def exp(self, data: NDArray[np.int64]) -> pd.DataFrame:
        result = super().exp(data)
        result["beta"] = self.beta
        result["frequency_encoding_length"] = self.frequency_encoding_length
        result["residual_encoding_length"] = self.residual_encoding_length
        return result
