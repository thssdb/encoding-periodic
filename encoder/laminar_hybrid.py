import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple, List
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from encoder import Encoder
from encoder.laminar import LaminarEncoder
from custom_tools.run_length_encode import run_length_encode


class LaminarHybridEncoder(Encoder):
    def __init__(
        self, group_max_length: int = 2**10 - 1, min_beta: int = 5, max_beta: int = 20
    ):
        super().__init__()
        self.group_max_length = group_max_length
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.laminar_encoder_dense = LaminarEncoder(
            sparse_mode=False, group_max_length=group_max_length
        )
        self.laminar_encoder_sparse = LaminarEncoder(
            sparse_mode=True, group_max_length=group_max_length
        )

    def add_dense_mode(
        self,
        result_d2: NDArray[np.int64],
        real_bit_width: int,
        group_bit_width: int,
        subtract: bool = False,
    ) -> None:
        d = -1 if subtract else 1
        if self.min_beta < group_bit_width:
            result_d2[0] += d * (1 + group_bit_width - self.min_beta)
            if self.min_beta + 1 <= self.max_beta:
                result_d2[1] += d * (-1 - (1 + group_bit_width - self.min_beta))
            if group_bit_width <= self.max_beta:
                result_d2[group_bit_width - self.min_beta] += d * (-1)
            if group_bit_width + 1 <= self.max_beta:
                result_d2[group_bit_width + 1 - self.min_beta] += d * 2

    def add_sparse_mode(
        self,
        result_d2: NDArray[np.int64],
        real_bit_width: int,
        group_bit_width: int,
        index_bit_length: int,
        subtract: bool = False,
    ) -> None:
        d = -1 if subtract else 1
        if self.min_beta < real_bit_width:
            result_d2[0] += d * (1 + group_bit_width - self.min_beta + index_bit_length)
            if self.min_beta + 1 <= self.max_beta:
                result_d2[1] += d * (
                    -1 - (1 + group_bit_width - self.min_beta + index_bit_length)
                )
            if real_bit_width <= self.max_beta:
                result_d2[real_bit_width - self.min_beta] += d * (
                    1 - (1 + group_bit_width - (real_bit_width - 1) + index_bit_length)
                )
            if real_bit_width + 1 <= self.max_beta:
                result_d2[real_bit_width + 1 - self.min_beta] += d * (
                    1 + group_bit_width - (real_bit_width - 1) + index_bit_length
                )

    def calculate_result(self, result_d2: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.cumsum(np.cumsum(result_d2, axis=0), axis=0)

    def estimate_value(
        self,
        data: NDArray[np.int64],
        bit_width: NDArray[np.int64],
        bit_width_suffix_max: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        n = data.shape[0]

        result_d2 = np.zeros(self.max_beta - self.min_beta + 1, dtype=np.int64)
        for i in range(n):
            self.add_sparse_mode(
                result_d2=result_d2,
                real_bit_width=bit_width[i],
                group_bit_width=bit_width_suffix_max[i],
                index_bit_length=(n - 1).bit_length(),
            )

        result = np.zeros(self.max_beta - self.min_beta + 1, dtype=np.int64)
        result.fill(np.iinfo(np.int64).max)
        k = min(n, 2 * (self.max_beta - self.min_beta + 1))
        for i in range(n):
            if i % k == 0:
                current_result = self.calculate_result(result_d2)
                result = np.minimum(result, current_result)
            self.add_sparse_mode(
                result_d2=result_d2,
                real_bit_width=bit_width[i],
                group_bit_width=bit_width_suffix_max[i],
                index_bit_length=(n - 1).bit_length(),
                subtract=True,
            )
            self.add_dense_mode(
                result_d2=result_d2,
                real_bit_width=bit_width[i],
                group_bit_width=bit_width_suffix_max[i],
            )
        current_result = self.calculate_result(result_d2)
        result = np.minimum(result, current_result)

        return result

    def estimate_length(
        self,
        data: NDArray[np.int64],
        bit_width: NDArray[np.int64],
        bit_width_suffix_max: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        n = data.shape[0]
        if n == 0:
            return np.array([], dtype=np.int64)

        group_size = 1 + self.group_max_length.bit_length()

        (
            bit_width_suffix_max_run_length_values,
            bit_width_suffix_max_run_length_counts,
        ) = run_length_encode(bit_width_suffix_max)

        result = np.zeros(self.max_beta - self.min_beta + 1, dtype=np.int64)
        for bit_width, count in zip(
            bit_width_suffix_max_run_length_values,
            bit_width_suffix_max_run_length_counts,
        ):
            for beta in range(self.min_beta, self.max_beta + 1):
                if bit_width > beta:
                    group_count = (
                        1 + (count + self.group_max_length - 1) // self.group_max_length
                    )
                    result[beta - self.min_beta] += group_count * group_size

        return result

    def partition(
        self,
        data: NDArray[np.int64],
    ) -> int:
        n = data.shape[0]

        bit_width = np.ceil(np.log2(np.abs(data) + 1)).astype(np.int64)
        bit_width_suffix_max = np.flip(
            np.maximum.accumulate(np.flip(bit_width))
        ).astype(np.int64)
        index_bit_length = (n - 1).bit_length()

        current_length = 0
        for i in range(n):
            if data[i] != 0:
                current_length += 1 + bit_width_suffix_max[i] + index_bit_length

        best_p = 0
        best_length = current_length

        for i in range(n):
            if data[i] != 0:
                current_length -= 1 + bit_width_suffix_max[i] + index_bit_length
            current_length += 1 + bit_width_suffix_max[i]
            if current_length < best_length:
                best_length = current_length
                best_p = i + 1

        return best_p

    def encode_stream_param(
        self, stream: BitStream, data: NDArray[np.int64], save_length: bool = True
    ) -> None:
        n = data.shape[0]
        if save_length:
            stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))

        if n == 0:
            return

        partition_index = self.partition(data)
        stream.append(BitArray(uint=partition_index, length=n.bit_length()))

        if partition_index > 0:
            self.laminar_encoder_dense.encode_stream_param(
                stream,
                data[:partition_index],
                save_length=False,
            )
        if partition_index < n:
            self.laminar_encoder_sparse.encode_stream_param(
                stream,
                data[partition_index:],
                save_length=False,
            )

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]) -> None:
        self.encode_stream_param(stream, data, save_length=True)

    def decode_param(
        self, stream: BitStream, save_length: bool = True, given_length: int = 0
    ) -> NDArray[np.int64]:
        if save_length:
            n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        else:
            n = given_length
        if n == 0:
            return np.array([], dtype=np.int64)

        partition_index = int(stream.read(n.bit_length()).uint)

        data = np.zeros(n, dtype=np.int64)
        if partition_index > 0:
            data[:partition_index] = self.laminar_encoder_dense.decode_param(
                stream, save_length=False, given_length=partition_index
            )
        if partition_index < n:
            data[partition_index:] = self.laminar_encoder_sparse.decode_param(
                stream, save_length=False, given_length=n - partition_index
            )

        return data

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        return self.decode_param(stream, save_length=True)

    def get_name(self) -> str:
        return "Laminar-Hybrid"
