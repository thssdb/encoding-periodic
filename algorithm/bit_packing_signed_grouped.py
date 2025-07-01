import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.bit_packing_signed import (
    bit_packing_signed_encode,
    bit_packing_signed_decode,
)


def group_size():
    return 2**8


def bit_packing_signed_grouped_encode(
    stream: BitStream,
    data: NDArray[np.int64],
    save_length: bool = True,
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return

    for i in range(0, n, group_size()):
        group = data[i : min(i + group_size(), n)]
        bit_packing_signed_encode(stream, group, save_length=False)


def bit_packing_signed_grouped_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.int64]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length
    if n == 0:
        return np.array([], dtype=np.int64)

    data = []
    for i in range(0, n, group_size()):
        group_length = min(group_size(), n - i)
        group = bit_packing_signed_decode(
            stream, save_length=False, given_length=group_length
        )
        data.append(group)

    return np.concatenate(data).astype(np.int64)
