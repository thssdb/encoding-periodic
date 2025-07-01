import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
import math


class BuffEncoder(Encoder):

    bits_needed = [0, 5, 8, 11, 15, 18, 21, 25, 28, 31, 35]

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        stream.append(BitArray(uint=len(data), length=32))
        precision = 0
        stream.append(BitArray(uint=precision, length=32))
        if len(data) == 0:
            return
        frac_bit = math.ceil(math.log2(10**precision))
        if precision >= 0 and precision < len(self.bits_needed):
            frac_bit = self.bits_needed[precision]
        min_value: int = math.floor(data[0] / (10**precision))
        max_value: int = math.floor(data[0] / (10**precision))
        for value in data:
            min_value = min(min_value, math.floor(value / (10**precision)))
            max_value = max(max_value, math.floor(value / (10**precision)))
        int_bit = math.ceil(math.log2(max_value - min_value + 1))
        stream.append(BitArray(int=min_value, length=32))
        stream.append(BitArray(uint=int_bit, length=32))
        bytes_group: list[BitStream] = []
        for value in data:
            bytes_group.append(BitStream())
            tmp = value // (10**precision)
            if int_bit > 0:
                bytes_group[-1].append(BitArray(uint=tmp - min_value, length=int_bit))
            frac_part_int = (value - min_value * (10**precision)) % (10**precision)
            frac_part = frac_part_int * (2**frac_bit) // (10**precision)
            if frac_bit > 0:
                bytes_group[-1].append(BitArray(uint=frac_part, length=frac_bit))
        for i in range(len(bytes_group)):
            bytes_group[i].pos = 0
        for j in range(int((int_bit + frac_bit + 7) / 8)):
            for i in range(len(bytes_group)):
                stream.append(
                    BitArray(bytes_group[i].read(min(8, int_bit + frac_bit - j * 8)))
                )

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        result = []
        result_len = int(stream.read(32).uint)
        precision = int(stream.read(32).uint)
        if result_len == 0:
            return np.asarray([], dtype=np.int64)
        frac_bit = math.ceil(math.log2(10**precision))
        if precision >= 0 and precision < len(self.bits_needed):
            frac_bit = self.bits_needed[precision]
        min_value = int(stream.read(32).int)
        int_bit = int(stream.read(32).uint)
        bytes_group: list[BitStream] = []
        for _ in range(result_len):
            bytes_group.append(BitStream())
        for j in range(int((int_bit + frac_bit + 7) / 8)):
            for i in range(result_len):
                bytes_group[i].append(
                    BitArray(stream.read(min(8, int_bit + frac_bit - j * 8)))
                )
        for i in range(result_len):
            stream_single = bytes_group[i]
            stream_single.pos = 0
            int_part = int(stream_single.read(int_bit).uint) if int_bit > 0 else 0
            frac_part = int(stream_single.read(frac_bit).uint) if frac_bit > 0 else 0
            frac_part_int = math.ceil(frac_part * (10**precision) / (2**frac_bit))
            value = (
                int_part * (10**precision) + frac_part_int + min_value * (10**precision)
            )
            value = value % (1 << 32)
            if value >= 1 << 31:
                value -= 1 << 32
            result.append(value)
        return np.asarray(result, dtype=np.int64)

    def get_name(self) -> str:
        return "Buff"
