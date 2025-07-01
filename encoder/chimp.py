import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder

PREVIOUS_VALUES = 128
PREVIOUS_VALUES_LOG2 = int(np.log2(PREVIOUS_VALUES))
THRESHOLD = 6 + PREVIOUS_VALUES_LOG2
SET_LSB = 2 ** (THRESHOLD + 1) - 1

LEADING_REPRESENTATION = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
]

LEADING_ROUND = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    8,
    8,
    8,
    8,
    12,
    12,
    12,
    12,
    16,
    16,
    18,
    18,
    20,
    20,
    22,
    22,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
]

stored_values: list[int] = []
indices: list[int] = []
index = 0
stored_leading_zeros = 0


LEADING_REPRESENTATION_DECODING = [0, 8, 12, 16, 18, 20, 22, 24]


def compress_value(stream: BitStream, value: int):
    value = int(value)
    if value < 0:
        value += 1 << 32
    global index, stored_leading_zeros
    previous_index = 0
    xor = 0
    trailing_zeros = 0
    key = value & SET_LSB
    cur_index = indices[key]
    if cur_index != -1 and (index - cur_index) < PREVIOUS_VALUES:
        tmp_xor = value ^ stored_values[cur_index % PREVIOUS_VALUES]
        trailing_zeros = trailingzero(tmp_xor)
        if trailing_zeros > THRESHOLD:
            previous_index = cur_index % PREVIOUS_VALUES
            xor = tmp_xor
        else:
            previous_index = index % PREVIOUS_VALUES
            xor = stored_values[previous_index] ^ value
    else:
        previous_index = index % PREVIOUS_VALUES
        xor = stored_values[previous_index] ^ value
    if xor == 0:
        stream.append(BitArray(uint=0, length=2))
        stream.append(BitArray(uint=previous_index, length=PREVIOUS_VALUES_LOG2))
        stored_leading_zeros = 32 + 1
    else:
        leading_zeros = LEADING_ROUND[leadingzero(xor)]
        if trailing_zeros > THRESHOLD:
            stream.append(BitArray(uint=1, length=2))
            stream.append(BitArray(uint=previous_index, length=PREVIOUS_VALUES_LOG2))
            stream.append(
                BitArray(uint=LEADING_REPRESENTATION[leading_zeros], length=3)
            )
            significant_bits = 32 - leading_zeros - trailing_zeros
            stream.append(BitArray(uint=significant_bits, length=6))
            if significant_bits > 0:
                stream.append(
                    BitArray(uint=xor >> trailing_zeros, length=significant_bits)
                )
            stored_leading_zeros = 32 + 1
        elif leading_zeros == stored_leading_zeros:
            stream.append(BitArray(uint=2, length=2))
            significant_bits = 32 - leading_zeros
            if significant_bits > 0:
                stream.append(BitArray(uint=xor, length=significant_bits))
        else:
            stream.append(BitArray(uint=3, length=2))
            stored_leading_zeros = leading_zeros
            stream.append(
                BitArray(uint=LEADING_REPRESENTATION[leading_zeros], length=3)
            )
            significant_bits = 32 - leading_zeros
            if significant_bits > 0:
                stream.append(BitArray(uint=xor, length=significant_bits))
    index += 1
    indices[key] = index
    stored_values[index % PREVIOUS_VALUES] = value


def decompress_value(stream: BitStream) -> int:
    global index, stored_values, stored_leading_zeros
    result = 0
    previous_index = 0
    cur_type = int(stream.read(2).uint)
    if cur_type == 0:
        previous_index = int(stream.read(PREVIOUS_VALUES_LOG2).uint)
        result = stored_values[previous_index]
        stored_leading_zeros = 32 + 1
    elif cur_type == 1:
        previous_index = int(stream.read(PREVIOUS_VALUES_LOG2).uint)
        leading_zeros = LEADING_REPRESENTATION_DECODING[int(stream.read(3).uint)]
        significant_bits = int(stream.read(6).uint)
        trailing_zeros = 32 - leading_zeros - significant_bits
        result = stored_values[previous_index] ^ (
            (int(stream.read(significant_bits).uint) if significant_bits > 0 else 0)
            << trailing_zeros
        )
        stored_leading_zeros = 32 + 1
    elif cur_type == 2:
        previous_index = index % PREVIOUS_VALUES
        leading_zeros = stored_leading_zeros
        significant_bits = 32 - leading_zeros
        result = stored_values[previous_index] ^ (
            int(stream.read(significant_bits).uint) if significant_bits > 0 else 0
        )
    else:
        previous_index = index % PREVIOUS_VALUES
        leading_zeros = LEADING_REPRESENTATION_DECODING[int(stream.read(3).uint)]
        stored_leading_zeros = leading_zeros
        significant_bits = 32 - leading_zeros
        result = stored_values[previous_index] ^ (
            int(stream.read(significant_bits).uint) if significant_bits > 0 else 0
        )
    index += 1
    stored_values[index % PREVIOUS_VALUES] = result
    return result


def leadingzero(x: int) -> int:
    x = int(x)
    if x < 0:
        return 0
    elif x == 0:
        return 32
    return 32 - x.bit_length()


def trailingzero(x: int) -> int:
    x = int(x)
    if x == 0:
        return 32
    return (x - (x & (x - 1))).bit_length() - 1


class ChimpEncoder(Encoder):
    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        global index, indices, stored_values, stored_leading_zeros
        stored_leading_zeros = 0
        indices = [-1] * (2 ** (THRESHOLD + 1))
        stored_values = [0] * PREVIOUS_VALUES
        stream.append(BitArray(uint=len(data), length=32))
        stream.append(
            BitArray(
                uint=int(data[0]) if data[0] >= 0 else int(data[0]) + (1 << 32),
                length=32,
            )
        )
        index = 0
        indices[data[0] & SET_LSB] = index
        stored_values[0] = int(data[0]) if data[0] >= 0 else int(data[0]) + (1 << 32)
        for i in range(1, len(data)):
            compress_value(stream, data[i])

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        global index, stored_values, stored_leading_zeros
        stored_leading_zeros = 0
        stored_values = [0] * PREVIOUS_VALUES
        length = int(stream.read(32).uint)
        first_value = int(stream.read(32).uint)
        result = [first_value]
        index = 0
        stored_values[0] = first_value
        for i in range(1, length):
            result.append(decompress_value(stream))

        for i in range(length):
            if result[i] >= 1 << 31:
                result[i] -= 1 << 32
        return np.asarray(result, dtype=np.int64)

    def get_name(self) -> str:
        return "Chimp"
