import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.utils import get_bit_length_cnt
from algorithm.descending_bit_packing import (
    descending_bit_packing_encode,
    descending_bit_packing_decode,
)
from algorithm.bit_packing import bit_packing_encode, bit_packing_decode
from algorithm.sign_packing import sign_packing_encode, sign_packing_decode


def get_optimal_d(n: int, cnt: NDArray[np.int64]) -> int:
    optimal_result = 0
    optimal_d = -1
    current_count = 0
    current_bit = 0
    for d in range(len(cnt) - 1, -1, -1):
        result = current_bit + (n - current_count) * d
        if optimal_d == -1 or result < optimal_result:
            optimal_result = result
            optimal_d = d
        current_count += cnt[d]
        current_bit += (d + (n - 1).bit_length()) * cnt[d]
    return optimal_d


def separate_storage_encode(
    stream: BitStream,
    data: NDArray[np.int64],
    save_length: bool = True,
    is_given_d: bool = False,
    given_d: int = 0,
) -> int:
    n = data.shape[0]
    sign_array = np.zeros(n, dtype=np.bool_)
    data_unsigned = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        if data[i] < 0:
            sign_array[i] = True
            data_unsigned[i] = -int(data[i])
        else:
            data_unsigned[i] = int(data[i])

    cnt = get_bit_length_cnt(data_unsigned)
    optimal_d = get_optimal_d(n, cnt) if not is_given_d else given_d

    stream.append(BitArray(uint=optimal_d, length=MAX_VALUE.bit_length()))
    high_bit_array = (data_unsigned >> optimal_d).astype(np.uint64)
    low_bit_array = (data_unsigned & ((1 << optimal_d) - 1)).astype(np.uint64)
    sign_packing_encode(stream, sign_array, save_length=save_length)
    if optimal_d > 0:
        bit_packing_encode(stream, low_bit_array, optimal_d, save_length=False)
    descending_bit_packing_encode(stream, high_bit_array, save_length=False)
    return optimal_d


def separate_storage_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.int64]:
    optimal_d = int(stream.read(MAX_VALUE.bit_length()).uint)
    sign_array = sign_packing_decode(
        stream, save_length=save_length, given_length=given_length
    )
    n = sign_array.shape[0]
    if optimal_d > 0:
        low_bit_array = bit_packing_decode(stream, save_length=False, given_length=n)
    else:
        low_bit_array = np.zeros(n, dtype=np.uint64)
    high_bit_array = descending_bit_packing_decode(
        stream, save_length=False, given_length=n
    )
    data_unsigned = ((high_bit_array << optimal_d) | low_bit_array).astype(np.uint64)
    data = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if sign_array[i]:
            data[i] = -int(data_unsigned[i])
        else:
            data[i] = int(data_unsigned[i])
    return data
