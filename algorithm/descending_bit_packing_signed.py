import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.utils import get_bit_length_cnt


def descending_bit_packing_signed_encode(
    stream: BitStream, data: NDArray[np.int64], save_length: bool = True
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return
    cnt = get_bit_length_cnt(np.abs(data).astype(np.uint64))
    m = n - cnt[0]
    stream.append(BitArray(uint=m, length=MAX_LENGTH.bit_length()))
    if m == 0:
        return
    for i in range(len(cnt) - 2, 0, -1):
        cnt[i] += cnt[i + 1]
    index_sorted = np.zeros(m, dtype=np.uint64)
    value_sorted = np.zeros(m, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        if data[i] != 0:
            pos = int(np.abs(data[i])).bit_length()
            cnt[pos] -= 1
            index_sorted[cnt[pos]] = i
            value_sorted[cnt[pos]] = data[i]
    for i in range(m):
        stream.append(
            BitArray(uint=index_sorted[i], length=max(1, (n - 1).bit_length()))
        )
    current_length = int(np.abs(value_sorted[0])).bit_length()
    stream.append(BitArray(uint=current_length, length=MAX_VALUE.bit_length()))
    for i in range(m):
        stream.append(BitArray(int=value_sorted[i], length=current_length + 1))
        current_length = int(np.abs(value_sorted[i])).bit_length()
    return


def descending_bit_packing_signed_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.int64]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length
    if n == 0:
        return np.array([], dtype=np.int64)
    m = int(stream.read(MAX_LENGTH.bit_length()).uint)
    if m == 0:
        return np.zeros(n, dtype=np.int64)
    data = np.zeros(n, dtype=np.int64)
    index_sorted = np.zeros(m, dtype=np.uint64)
    for i in range(m):
        index_sorted[i] = int(stream.read(max(1, (n - 1).bit_length())).uint)
    current_length = int(stream.read(MAX_VALUE.bit_length()).uint)
    for i in range(m):
        value = int(stream.read(current_length + 1).int)
        data[index_sorted[i]] = value
        current_length = int(np.abs(value)).bit_length()
    return data
