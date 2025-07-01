import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH


def sign_packing_encode(
    stream: BitStream, data: NDArray[np.bool_], save_length: bool = True
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return
    for i in range(n):
        stream.append(BitArray(uint=int(data[i]), length=1))


def sign_packing_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.bool_]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length
    if n == 0:
        return np.array([], dtype=np.bool_)
    data = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        data[i] = bool(stream.read(1).uint)
    return data
