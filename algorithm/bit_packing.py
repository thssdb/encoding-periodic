import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE


def bit_packing_encode(
    stream: BitStream,
    data: NDArray[np.uint64],
    bit_length: int,
    save_length: bool = True,
):
    """
    Encodes an array of uint64 values into a bit stream using a specified bit length.

    Parameters:
    - stream: BitStream to write the encoded data.
    - data: NDArray of uint64 values to encode.
    - bit_length: Number of bits to use for each value in the data array.
    """
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return
    stream.append(BitArray(uint=bit_length, length=MAX_VALUE.bit_length()))
    if bit_length == 0:
        return
    for i in range(n):
        stream.append(BitArray(uint=data[i], length=bit_length))


def bit_packing_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.uint64]:
    """
    Decodes a bit stream into an array of uint64 values using a specified bit length.

    Parameters:
    - stream: BitStream to read the encoded data from.
    - bit_length: Number of bits used for each value in the data array.

    Returns:
    - NDArray of uint64 values decoded from the bit stream.
    """
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length
    if n == 0:
        return np.array([], dtype=np.uint64)
    bit_length = int(stream.read(MAX_VALUE.bit_length()).uint)
    if bit_length == 0:
        return np.zeros(n, dtype=np.uint64)
    data = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        data[i] = int(stream.read(bit_length).uint)
    return data
