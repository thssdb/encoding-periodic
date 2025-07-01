import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from algorithm.bit_packing import bit_packing_encode, bit_packing_decode
from encoder import Encoder


class TS2DiffEncoder(Encoder):
    def block_size(self) -> int:
        return 2**8

    def encode_stream_block(self, stream: BitStream, data: NDArray[np.int64]):
        first_value = int(data[0])
        stream.append(BitStream(int=first_value, length=MAX_VALUE.bit_length()))
        if data.shape[0] == 1:
            return
        diff_data = np.diff(data)
        diff_min = int(np.min(diff_data))
        stream.append(BitStream(int=diff_min, length=MAX_VALUE.bit_length()))
        diff2_data = (diff_data - diff_min).astype(np.uint64)
        diff2_data_max_bit_length = int(diff2_data.max()).bit_length()
        bit_packing_encode(
            stream, diff2_data, diff2_data_max_bit_length, save_length=False
        )

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        n = data.shape[0]
        stream.append(BitStream(uint=n, length=MAX_LENGTH.bit_length()))
        if n == 0:
            return
        for i in range(0, n, self.block_size()):
            block_data = data[i : min(i + self.block_size(), n)]
            self.encode_stream_block(stream, block_data)
        return

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        n = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if n == 0:
            return np.array([], dtype=np.int64)
        data = np.zeros(n, dtype=np.int64)
        for i in range(0, n, self.block_size()):
            block_size = min(self.block_size(), n - i)
            first_value = int(stream.read(MAX_VALUE.bit_length()).int)
            if block_size == 1:
                data[i] = first_value
                continue
            diff_min = int(stream.read(MAX_VALUE.bit_length()).int)
            diff2_data = bit_packing_decode(
                stream, save_length=False, given_length=block_size - 1
            )
            diff_data = diff2_data.astype(np.int64) + diff_min
            data[i] = first_value
            for j in range(1, block_size):
                data[i + j] = data[i + j - 1] + diff_data[j - 1]
        return data

    def get_name(self) -> str:
        return "TS-2-Diff"
