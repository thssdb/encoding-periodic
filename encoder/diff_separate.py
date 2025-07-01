import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from algorithm.constants import MAX_LENGTH
from algorithm.separate_storage import (
    separate_storage_encode,
    separate_storage_decode,
)
from encoder import Encoder


class DiffSeparateEncoder(Encoder):
    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        n = data.shape[0]
        stream.append(BitStream(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return
        diff_data = np.concatenate(([data[0]], np.diff(data)))
        separate_storage_encode(stream, diff_data, save_length=False)
        return

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)
        diff_data = separate_storage_decode(stream, save_length=False, given_length=n)
        data = np.cumsum(diff_data)
        return data

    def get_name(self) -> str:
        return "Diff-Separate"
