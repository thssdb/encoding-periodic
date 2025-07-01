import numpy as np
import pandas as pd
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.separate_storage import separate_storage_encode, separate_storage_decode
from encoder.laminar_hybrid_ablation import LaminarHybridEncoder
from algorithm.bit_packing_signed_grouped import (
    bit_packing_signed_grouped_decode,
    bit_packing_signed_grouped_encode,
)
from algorithm.descending_bit_packing_signed import (
    descending_bit_packing_signed_encode,
    descending_bit_packing_signed_decode,
)
import encoder.flea
import encoder.laminar_hybrid
from typing import Tuple


class FLEAEncoder(encoder.flea.FLEAEncoder):
    def init_laminar_hybrid_encoder(
        self,
    ) -> encoder.laminar_hybrid.LaminarHybridEncoder:
        return LaminarHybridEncoder(
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            is_given_p=self.is_given_p,
            given_p=self.given_p,
        )

    def __init__(
        self,
        min_beta: int = 5,
        max_beta: int = 20,
        is_given_beta: bool = False,
        given_beta: int = 0,
        is_given_p: bool = False,
        given_p: float = 0,
        simple_frequency: bool = False,
        simple_residual: bool = False,
        calculate_partition: bool = False,
        is_given_d: bool = False,
        given_d: int = 0,
    ):
        self.is_given_beta = is_given_beta
        self.given_beta = given_beta
        self.is_given_p = is_given_p
        self.given_p = given_p
        self.simple_frequency = simple_frequency
        self.simple_residual = simple_residual
        self.frequency_encoding_length = -1
        self.residual_encoding_length = -1
        self.low_bit_encoding_length = -1
        self.calculate_partition = calculate_partition
        self.partition_prefix_encoding_length = np.asarray([], dtype=np.int64)
        self.partition_suffix_encoding_length = np.asarray([], dtype=np.int64)
        self.is_given_d = is_given_d
        self.given_d = given_d
        super().__init__(min_beta=min_beta, max_beta=max_beta)

    def encode_frequency(
        self,
        stream: BitStream,
        frequency_value_quantized_real: NDArray[np.int64],
        frequency_value_quantized_imag: NDArray[np.int64],
    ):
        start = stream.pos
        if self.simple_frequency:
            descending_bit_packing_signed_encode(
                stream, frequency_value_quantized_real, save_length=False
            )
            descending_bit_packing_signed_encode(
                stream, frequency_value_quantized_imag, save_length=False
            )
        else:
            super().encode_frequency(
                stream, frequency_value_quantized_real, frequency_value_quantized_imag
            )
        if self.calculate_partition and isinstance(
            self.laminar_hybrid_encoder, LaminarHybridEncoder
        ):
            (
                self.partition_prefix_encoding_length,
                self.partition_suffix_encoding_length,
            ) = self.laminar_hybrid_encoder.get_partition_array(
                frequency_value_quantized_imag
            )
        self.frequency_encoding_length = stream.pos - start

    def encode_residual(
        self,
        stream: BitStream,
        residual: NDArray[np.int64],
    ):
        start = stream.pos
        if self.given_d:
            separate_storage_encode(
                stream,
                residual,
                save_length=False,
                is_given_d=self.is_given_d,
                given_d=self.given_d,
            )
            self.low_bit_encoding_length = len(residual) * (1 + self.given_d)
        elif self.simple_residual:
            bit_packing_signed_grouped_encode(stream, residual, save_length=False)
        else:
            super().encode_residual(stream, residual)
        self.residual_encoding_length = stream.pos - start

    def decode_frequency(
        self, stream: BitStream, n: int
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        if self.simple_frequency:
            frequency_value_quantized_real = descending_bit_packing_signed_decode(
                stream, save_length=False, given_length=n // 2 + 1
            )
            frequency_value_quantized_imag = descending_bit_packing_signed_decode(
                stream, save_length=False, given_length=n // 2 + 1
            )
        else:
            frequency_value_quantized_real, frequency_value_quantized_imag = (
                super().decode_frequency(stream, n)
            )
        return (
            frequency_value_quantized_real,
            frequency_value_quantized_imag,
        )

    def decode_residual(self, stream: BitStream, n: int) -> NDArray[np.int64]:
        if self.simple_residual:
            residual = bit_packing_signed_grouped_decode(
                stream, save_length=False, given_length=n
            )
        else:
            residual = super().decode_residual(stream, n)
        return residual

    def get_optimal_beta(
        self,
        n: int,
        frequency_value_real: NDArray[np.int64],
        frequency_value_imag: NDArray[np.int64],
    ):
        if self.is_given_beta:
            return np.int64(self.given_beta)
        else:
            return super().get_optimal_beta(
                n, frequency_value_real, frequency_value_imag
            )

    def get_name(self) -> str:
        base_name = super().get_name()
        if self.is_given_beta:
            base_name += f"_beta{self.given_beta}"
        if self.is_given_p:
            base_name += f"_p{self.given_p}"
        if self.is_given_d:
            base_name += f"_d{self.given_d}"
        if self.simple_frequency:
            base_name += "_simple_frequency"
        if self.simple_residual:
            base_name += "_simple_residual"
        return base_name

    def exp(self, data: NDArray[np.int64]) -> pd.DataFrame:
        result = super().exp(data)
        result["is_given_beta"] = self.is_given_beta
        result["given_beta"] = self.given_beta
        result["is_given_p"] = self.is_given_p
        result["given_p"] = self.given_p
        result["is_given_d"] = self.is_given_d
        result["given_d"] = self.given_d
        result["simple_frequency"] = self.simple_frequency
        result["simple_residual"] = self.simple_residual
        result["frequency_encoding_length"] = self.frequency_encoding_length
        result["residual_encoding_length"] = self.residual_encoding_length
        result["low_bit_encoding_length"] = self.low_bit_encoding_length
        return result
