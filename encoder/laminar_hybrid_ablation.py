import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple, List
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from encoder import Encoder
from encoder.laminar import LaminarEncoder
from custom_tools.run_length_encode import run_length_encode
import encoder.laminar_hybrid


class LaminarHybridEncoder(encoder.laminar_hybrid.LaminarHybridEncoder):
    def __init__(
        self,
        group_max_length: int = 2**10 - 1,
        min_beta: int = 5,
        max_beta: int = 20,
        is_given_p: bool = False,
        given_p: float = 0,
    ):
        super().__init__(
            group_max_length=group_max_length,
            min_beta=min_beta,
            max_beta=max_beta,
        )
        self.is_given_p = is_given_p
        self.given_p = given_p

    def partition(
        self,
        data: NDArray[np.int64],
    ) -> int:
        if self.is_given_p:
            partition_index = int(self.given_p * data.shape[0])
        else:
            partition_index = super().partition(data)
        return partition_index

    def get_name(self) -> str:
        base_name = super().get_name()

        if self.is_given_p:
            base_name += f"-p{self.given_p}"

        return base_name

    def get_partition_array(
        self,
        data: NDArray[np.int64],
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        n = data.shape[0]

        bit_width = np.ceil(np.log2(np.abs(data) + 1)).astype(np.int64)
        bit_width_suffix_max = np.flip(
            np.maximum.accumulate(np.flip(bit_width))
        ).astype(np.int64)
        index_bit_length = (n - 1).bit_length()

        current_prefix_length = 0
        current_suffix_length = 0
        for i in range(n):
            if data[i] != 0:
                current_suffix_length += 1 + bit_width_suffix_max[i] + index_bit_length

        prefix_encoding_length = np.zeros(n + 1, dtype=np.int64)
        suffix_encoding_length = np.zeros(n + 1, dtype=np.int64)
        prefix_encoding_length[0] = current_prefix_length
        suffix_encoding_length[0] = current_suffix_length

        for i in range(n):
            if data[i] != 0:
                current_suffix_length -= 1 + bit_width_suffix_max[i] + index_bit_length
            current_prefix_length += 1 + bit_width_suffix_max[i]
            prefix_encoding_length[i + 1] = current_prefix_length
            suffix_encoding_length[i + 1] = current_suffix_length

        return prefix_encoding_length, suffix_encoding_length
