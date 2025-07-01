import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple, List
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from encoder import Encoder
from encoder.rle import RLEEncoder
from custom_tools.run_length_encode import run_length_encode


class LaminarEncoder(Encoder):
    def __init__(self, sparse_mode: bool = False, group_max_length: int = 2**10 - 1):
        super().__init__()
        self.sparse_mode = sparse_mode
        self.rle_sub_encoder = RLEEncoder(
            max_length=group_max_length,
            max_value=1,
            signed=False,
            special_mode=True,
            special_mode_value=1,
            special_mode_max_length=7,
        )

    def encode_stream_param(
        self, stream: BitStream, data: NDArray[np.int64], save_length: bool = True
    ):
        n = data.shape[0]
        if save_length:
            stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return

        non_zero_indices = np.asarray([], dtype=np.int64)
        if self.sparse_mode:
            non_zero_indices = np.nonzero(data)[0]
            data = data[non_zero_indices]

        bit_width = np.ceil(np.log2(np.abs(data) + 1)).astype(np.int64)
        bit_width_suffix_max = np.flip(
            np.maximum.accumulate(np.flip(bit_width))
        ).astype(np.int64)
        (
            bit_width_suffix_max_run_length_values,
            bit_width_suffix_max_run_length_counts,
        ) = run_length_encode(bit_width_suffix_max)

        instructions: List[int] = []
        last_bit_width = (
            0
            if len(bit_width_suffix_max_run_length_values) == 0
            else bit_width_suffix_max_run_length_values[0]
        )
        stream.append(
            BitArray(uint=last_bit_width, length=MAX_VALUE.bit_length().bit_length())
        )
        for bit_width, count in zip(
            bit_width_suffix_max_run_length_values,
            bit_width_suffix_max_run_length_counts,
        ):
            if last_bit_width != -1:
                instructions.extend([1] * (last_bit_width - bit_width))
            last_bit_width = bit_width
            instructions.extend([0] * count)

        self.rle_sub_encoder.encode_stream(
            stream,
            np.array(instructions, dtype=np.int64),
        )

        m = len(data)

        for i in range(m):
            if self.sparse_mode:
                stream.append(
                    BitArray(uint=non_zero_indices[i], length=(n - 1).bit_length())
                )
            stream.append(
                BitArray(
                    int=data[i],
                    length=int(bit_width_suffix_max[i] + 1),
                )
            )

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        self.encode_stream_param(stream, data, save_length=True)

    def decode_param(
        self, stream: BitStream, save_length: bool = True, given_length: int = 0
    ) -> NDArray[np.int64]:
        if save_length:
            n = stream.read(MAX_LENGTH.bit_length()).uint
        else:
            n = given_length
        if n == 0:
            return np.array([], dtype=np.int64)

        bit_width = stream.read(MAX_VALUE.bit_length().bit_length()).uint

        instructions = self.rle_sub_encoder.decode(stream)

        m = len(instructions) - np.sum(instructions)
        bit_width_suffix_max = np.zeros(m, dtype=np.int64)
        bit_width_suffix_max_index = 0
        current_bit_width = bit_width

        for i in range(len(instructions)):
            if instructions[i] == 1:
                current_bit_width -= 1
            else:
                bit_width_suffix_max[bit_width_suffix_max_index] = current_bit_width
                bit_width_suffix_max_index += 1

        data = np.zeros(n, dtype=np.int64)
        for i in range(m):
            if self.sparse_mode:
                current_index = stream.read((n - 1).bit_length()).uint
            else:
                current_index = i
            data[current_index] = stream.read(bit_width_suffix_max[i] + 1).int

        return data

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        return self.decode_param(stream, save_length=True)

    def get_name(self) -> str:
        return "Laminar"
