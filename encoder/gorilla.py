import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream
from encoder import Encoder


def leadingzero(x: int) -> int:
    if x < 0:
        return 0
    elif x == 0:
        return 32
    return 32 - x.bit_length()


def trailingzero(x: int) -> int:
    if x == 0:
        return 32
    return (x - (x & (x - 1))).bit_length() - 1


class GorillaEncoder(Encoder):
    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        lastl = lastt = -1
        data = data.tolist()
        for i in range(len(data)):
            if data[i] < 0:
                data[i] += 1 << 32
        stream.append(BitStream(uint=data[0], length=32))
        stream.append(BitStream(uint=len(data), length=32))
        for i in range(1, len(data)):
            x = int(data[i] ^ data[i - 1])
            if x == 0:
                stream.append(BitStream(uint=0, length=1))
            else:
                l = leadingzero(x)
                t = trailingzero(x)
                if lastl != -1 and l >= lastl and t >= lastt:
                    stream.append(BitStream(uint=2, length=2))
                    stream.append(BitStream(uint=x >> lastt, length=32 - lastl - lastt))
                else:
                    stream.append(BitStream(uint=3, length=2))
                    stream.append(BitStream(uint=l, length=6))
                    stream.append(BitStream(uint=t, length=6))
                    stream.append(BitStream(uint=x >> t, length=32 - l - t))
                    lastl = l
                    lastt = t

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        first_value = int(stream.read(32).uint)
        length = int(stream.read(32).uint)
        result = [first_value]
        l = t = -1
        for i in range(1, length):
            type = int(stream.read(1).uint)
            if type == 0:
                result.append(result[-1])
            else:
                type = int(stream.read(1).uint)
                if type == 0:
                    x = int(stream.read(32 - l - t).uint)
                    result.append(result[-1] ^ (x << t))
                else:
                    l = int(stream.read(6).uint)
                    t = int(stream.read(6).uint)
                    x = int(stream.read(32 - l - t).uint)
                    result.append(result[-1] ^ (x << t))

        for i in range(length):
            if result[i] >= 1 << 31:
                result[i] -= 1 << 32
        return np.asarray(result, dtype=np.int64)

    def get_name(self) -> str:
        return "Gorilla"
