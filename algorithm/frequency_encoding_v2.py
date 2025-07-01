import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.variable_length_decreasing_encoding import get_suffix_length
from algorithm.variable_length_decreasing_encoding import (
    variable_length_decreasing_encode,
    variable_length_decreasing_decode,
    MAX_GROUP_SIZE,
)
from algorithm.descending_bit_packing_signed import (
    descending_bit_packing_signed_encode,
    descending_bit_packing_signed_decode,
)


def get_variable_length_decreasing_encoding_prefix_length(
    data: NDArray[np.int64], suffix_length: NDArray[np.uint64]
) -> NDArray[np.uint64]:
    n = data.shape[0]
    variable_length_decreasing_encoding_prefix_length = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        variable_length_decreasing_encoding_prefix_length[i] = (
            variable_length_decreasing_encoding_prefix_length[i - 1] if i > 0 else 0
        )
        if i == 0 or suffix_length[i] != suffix_length[i - 1]:
            variable_length_decreasing_encoding_prefix_length[i] += (
                MAX_VALUE.bit_length().bit_length() + MAX_GROUP_SIZE.bit_length()
            )
        variable_length_decreasing_encoding_prefix_length[i] += int(
            suffix_length[i] + 1
        )
    return variable_length_decreasing_encoding_prefix_length


def get_descending_bit_packing_suffix_length(
    data: NDArray[np.int64],
) -> NDArray[np.uint64]:
    n = data.shape[0]
    descending_bit_packing_suffix_length = np.zeros(n, dtype=np.uint64)
    non_zero_count = 0
    non_zero_bit_length_sum = 0
    for i in range(n - 1, -1, -1):
        if data[i] != 0:
            non_zero_count += 1
            non_zero_bit_length_sum += int(np.abs(data[i])).bit_length() + 1
        descending_bit_packing_suffix_length[i] = (
            max(1, (n - i).bit_length()) * non_zero_count + non_zero_bit_length_sum
        )
    return descending_bit_packing_suffix_length


def frequency_encode(
    stream: BitStream, data: NDArray[np.int64], save_length: bool = True
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return

    suffix_length, _ = get_suffix_length(data)

    variable_length_decreasing_encoding_prefix_length = (
        get_variable_length_decreasing_encoding_prefix_length(data, suffix_length)
    )

    descending_bit_packing_suffix_length = get_descending_bit_packing_suffix_length(
        data
    )

    optimal_pos = -1
    optimal_length = -1
    for i in range(n + 1):
        current_length = (
            variable_length_decreasing_encoding_prefix_length[i - 1] if i > 0 else 0
        ) + (descending_bit_packing_suffix_length[i] if i < n else 0)
        if optimal_pos == -1 or current_length < optimal_length:
            optimal_length = current_length
            optimal_pos = i

    stream.append(BitArray(uint=optimal_pos, length=MAX_LENGTH.bit_length()))
    if optimal_pos > 0:
        variable_length_decreasing_encode(
            stream,
            data[:optimal_pos],
            save_length=False,
        )
    if optimal_pos < n:
        descending_bit_packing_signed_encode(
            stream,
            data[optimal_pos:],
            save_length=False,
        )


def frequency_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.int64]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length

    if n == 0:
        return np.array([], dtype=np.int64)

    optimal_pos = int(stream.read(MAX_LENGTH.bit_length()).uint)
    data = np.zeros(n, dtype=np.int64)

    if optimal_pos > 0:
        data[:optimal_pos] = variable_length_decreasing_decode(
            stream, save_length=False, given_length=optimal_pos
        )
    if optimal_pos < n:
        data[optimal_pos:] = descending_bit_packing_signed_decode(
            stream, save_length=False, given_length=n - optimal_pos
        )

    return data
