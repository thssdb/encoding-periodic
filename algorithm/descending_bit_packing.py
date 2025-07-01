import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.utils import get_bit_length_cnt


def descending_bit_packing_encode(
    stream: BitStream, data: NDArray[np.uint64], save_length: bool = True
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return
    cnt = get_bit_length_cnt(data)
    m = n - cnt[0]
    stream.append(BitArray(uint=m, length=MAX_LENGTH.bit_length()))
    if m == 0:
        return
    for i in range(len(cnt) - 2, 0, -1):
        cnt[i] += cnt[i + 1]
    index_sorted = np.zeros(m, dtype=np.uint64)
    value_sorted = np.zeros(m, dtype=np.uint64)
    for i in range(n - 1, -1, -1):
        if data[i] != 0:
            cnt[int(data[i]).bit_length()] -= 1
            index_sorted[cnt[int(data[i]).bit_length()]] = i
            value_sorted[cnt[int(data[i]).bit_length()]] = data[i]
    for i in range(m):
        stream.append(BitArray(uint=index_sorted[i], length=(n - 1).bit_length()))
    current_length = int(value_sorted[0]).bit_length()
    stream.append(BitArray(uint=current_length, length=MAX_VALUE.bit_length()))
    for i in range(m):
        stream.append(BitArray(uint=value_sorted[i], length=current_length))
        current_length = int(value_sorted[i]).bit_length()
    return


def descending_bit_packing_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.uint64]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length
    if n == 0:
        return np.array([], dtype=np.uint64)
    m = int(stream.read(MAX_LENGTH.bit_length()).uint)
    if m == 0:
        return np.zeros(n, dtype=np.uint64)
    data = np.zeros(n, dtype=np.uint64)
    index_sorted = np.zeros(m, dtype=np.uint64)
    for i in range(m):
        index_sorted[i] = int(stream.read((n - 1).bit_length()).uint)
    current_length = int(stream.read(MAX_VALUE.bit_length()).uint)
    for i in range(m):
        value = int(stream.read(current_length).uint)
        data[index_sorted[i]] = value
        current_length = int(value).bit_length()
    return data
