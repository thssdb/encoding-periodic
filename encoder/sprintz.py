import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
from algorithm.constants import MAX_LENGTH, MAX_VALUE


class SprintzEncoder(Encoder):
    def block_size(self) -> int:
        return 2**8

    def encode_stream_block(self, stream: BitStream, data: NDArray[np.int64]):
        delta = np.diff(data)
        stream.append(BitArray(int=data[0], length=MAX_VALUE.bit_length()))
        stream.append(BitArray(uint=len(delta), length=MAX_LENGTH.bit_length()))
        if len(delta) == 0:
            return
        bit_len = 0
        for i in range(len(delta)):
            if delta[i] > 0:
                delta[i] = delta[i] * 2 - 1
            else:
                delta[i] = -delta[i] * 2
            bit_len = max(bit_len, int(delta[i]).bit_length())
        stream.append(BitArray(uint=bit_len, length=MAX_VALUE.bit_length()))
        for d in delta:
            if bit_len > 0:
                stream.append(BitArray(uint=d, length=bit_len))
        return

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        bcnt = len(data) // self.block_size()
        realBcnt = (len(data) + self.block_size() - 1) // self.block_size()
        stream.append(BitArray(uint=realBcnt, length=MAX_LENGTH.bit_length()))
        for i in range(bcnt):
            self.encode_stream_block(
                stream, data[i * self.block_size() : (i + 1) * self.block_size()]
            )
        if bcnt * self.block_size() < len(data):
            self.encode_stream_block(stream, data[bcnt * self.block_size() :])

    def decode_block(self, stream: BitStream) -> NDArray[np.int64]:
        first_value = int(stream.read(MAX_VALUE.bit_length()).int)
        delta_length = int(stream.read(MAX_LENGTH.bit_length()).uint)
        if delta_length == 0:
            return np.asarray([first_value], dtype=np.int64)
        bit_length = int(stream.read(MAX_VALUE.bit_length()).uint)
        delta = []
        for _ in range(delta_length):
            tmp = int(stream.read(bit_length).uint) if bit_length > 0 else 0
            if tmp % 2 == 0:
                tmp = -tmp // 2
            else:
                tmp = (tmp + 1) // 2
            delta.append(tmp)
        return np.cumsum([first_value] + delta).tolist()

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        result = []
        realBcnt = int(stream.read(MAX_LENGTH.bit_length()).uint)
        for tmp in range(realBcnt):
            result.extend(self.decode_block(stream))
        return np.asarray(result, dtype=np.int64)

    def get_name(self) -> str:
        return "Sprintz"
