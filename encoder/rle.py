import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
from custom_tools.run_length_encode import run_length_encode
from algorithm.constants import MAX_LENGTH, MAX_VALUE


class RLEEncoder(Encoder):
    def __init__(
        self,
        max_length: int = 2**6 - 1,
        max_value: int = MAX_VALUE,
        signed: bool = True,
        special_mode: bool = False,
        special_mode_value: int = 1,
        special_mode_max_length: int = 2**3 - 1,
    ):
        super().__init__()
        self.max_length = max_length
        self.max_length_length = self.max_length.bit_length()
        self.max_value = max_value
        self.max_value_length = self.max_value.bit_length()
        self.signed = signed
        self.special_mode = special_mode
        self.special_mode_value = special_mode_value
        self.special_mode_max_length = special_mode_max_length

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        values, counts = run_length_encode(data)
        groups_count = 0
        for value, count in zip(values, counts):
            current_max_length = (
                self.special_mode_max_length
                if self.special_mode and value == self.special_mode_value
                else self.max_length
            )
            groups_count += (count + current_max_length - 1) // current_max_length
        stream.append(BitArray(uint=groups_count, length=MAX_LENGTH.bit_length()))
        for value, count in zip(values, counts):
            current_max_length = (
                self.special_mode_max_length
                if self.special_mode and value == self.special_mode_value
                else self.max_length
            )
            current_max_length_length = current_max_length.bit_length()
            for i in range(0, count, current_max_length):
                repeat = min(current_max_length, count - i)
                if self.signed:
                    stream.append(BitArray(int=value, length=self.max_value_length + 1))
                else:
                    stream.append(BitArray(uint=value, length=self.max_value_length))
                stream.append(BitArray(uint=repeat, length=current_max_length_length))

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        groups_count = int(stream.read(MAX_LENGTH.bit_length()).uint)
        data = []
        for _ in range(groups_count):
            if self.signed:
                value = int(stream.read(self.max_value_length + 1).int)
            else:
                value = int(stream.read(self.max_value_length).uint)
            current_max_length = (
                self.special_mode_max_length
                if self.special_mode and value == self.special_mode_value
                else self.max_length
            )
            current_max_length_length = current_max_length.bit_length()
            repeat = int(stream.read(current_max_length_length).uint)
            data.extend([value] * repeat)
        return np.array(data, dtype=np.int64)

    def get_name(self) -> str:
        return "RLE"
