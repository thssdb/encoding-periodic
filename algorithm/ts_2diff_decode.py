import numpy as np
from mylib.byteToys import byteInToys


def decode_block(stream: byteInToys) -> list[int]:
    first_value = stream.decode(32)
    min_delta = stream.decode(32)
    if min_delta >= 1 << 31:
        min_delta = min_delta - (1 << 32)
    delta_length = stream.decode(8)
    if delta_length == 0:
        return [first_value]
    bit_length = stream.decode(8)
    delta = []
    for tmp in range(delta_length):
        delta.append(stream.decode(bit_length) + min_delta)
    return np.cumsum([first_value] + delta).tolist()


def ts_2diff_decode(file_path: str) -> list[int]:
    stream = byteInToys(file_path)
    result = []
    realBcnt = stream.decode(32)
    for tmp in range(realBcnt):
        result += decode_block(stream)
    for i in range(len(result)):
        if result[i] >= 1 << 31:
            result[i] -= 1 << 32
    return result


# print(ts_2diff_decode("result.bin"))
