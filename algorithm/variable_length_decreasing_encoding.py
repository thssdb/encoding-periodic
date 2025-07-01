import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.utils import get_bit_length_cnt
from typing import Tuple

MAX_GROUP_SIZE = 2**10 - 1


def get_suffix_length(
    data: NDArray[np.int64],
) -> Tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    n = data.shape[0]
    suffix_length = np.zeros(n, dtype=np.uint64)
    right_count = np.zeros(n, dtype=np.uint64)
    for i in range(n - 1, -1, -1):
        current_length = int(np.abs(data[i])).bit_length()
        (suffix_length[i], right_count[i]) = (
            (
                suffix_length[i + 1],
                right_count[i + 1] + 1 if right_count[i + 1] < MAX_GROUP_SIZE else 1,
            )
            if i < n - 1 and current_length <= suffix_length[i + 1]
            else (current_length, 1)
        )
    return suffix_length, right_count


def variable_length_decreasing_encode(
    stream: BitStream, data: NDArray[np.int64], save_length: bool = True
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return

    suffix_length, right_count = get_suffix_length(data)

    for i in range(n):
        if i == 0 or right_count[i - 1] == 1:
            stream.append(
                BitArray(
                    uint=suffix_length[i], length=MAX_VALUE.bit_length().bit_length()
                )
            )
            if suffix_length[i] == 0:
                break
            stream.append(
                BitArray(uint=right_count[i], length=MAX_GROUP_SIZE.bit_length())
            )
        stream.append(
            BitArray(
                int=int(data[i]),
                length=int(suffix_length[i] + 1),
            )
        )


def variable_length_decreasing_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.int64]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length

    if n == 0:
        return np.array([], dtype=np.int64)

    data = np.zeros(n, dtype=np.int64)
    idx = 0

    while True:
        suffix_length = int(stream.read(MAX_VALUE.bit_length().bit_length()).uint)
        if suffix_length == 0:
            break
        right_count = int(stream.read(MAX_GROUP_SIZE.bit_length()).uint)
        for _ in range(right_count):
            data[idx] = int(stream.read(suffix_length + 1).int)
            idx += 1
        if idx >= n:
            break

    return data
