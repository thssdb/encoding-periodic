import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.sign_packing import sign_packing_encode, sign_packing_decode
from algorithm.bit_packing import bit_packing_encode, bit_packing_decode


def bit_packing_signed_encode(
    stream: BitStream,
    data: NDArray[np.int64],
    save_length: bool = True,
):
    n = data.shape[0]
    if save_length:
        stream.append(BitArray(uint=n, length=MAX_LENGTH.bit_length()))
    if n == 0:
        return
    sign = (data < 0).astype(np.bool_)
    data_abs = np.abs(data).astype(np.uint64)

    bit_length = int(np.ceil(np.log2(data_abs.max() + 1)))
    sign_packing_encode(stream, sign, save_length=False)
    bit_packing_encode(stream, data_abs, bit_length, save_length=False)


def bit_packing_signed_decode(
    stream: BitStream, save_length: bool = True, given_length: int = 0
) -> NDArray[np.int64]:
    if save_length:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
    else:
        n = given_length
    if n == 0:
        return np.array([], dtype=np.int64)
    sign = sign_packing_decode(stream, save_length=False, given_length=n)
    data_abs = bit_packing_decode(stream, save_length=False, given_length=n)
    data = np.where(sign, -data_abs, data_abs).astype(np.int64)
    return data
