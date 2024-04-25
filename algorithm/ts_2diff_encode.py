import numpy as np
import pandas as pd
from mylib.byteToys import byteOutToys


block_size = 128
# have_first = False


def encode_block(stream: byteOutToys, data: list[int]) -> int:
    delta = np.diff(data)
    if len(delta) == 0:
        stream.encode(data[0], 32)
        stream.encode(0, 32)
        stream.encode(0, 8)
        return 32 + 32 + 8
    min_delta = np.min(delta)
    bit_len = int(max(np.max(delta) - min_delta, 1)).bit_length()
    # first_value + min_delta + len(delta) + bit_len + delta
    stream.encode(data[0], 32)
    stream.encode(min_delta, 32)
    stream.encode(len(delta), 8)
    stream.encode(bit_len, 8)
    for d in delta:
        stream.encode(d - min_delta, bit_len)
    return


def ts2diff_encode(data: list[int], file_path: str):
    stream = byteOutToys()
    bcnt = len(data) // block_size
    realBcnt = (len(data) + block_size - 1) // block_size
    stream.encode(realBcnt, 32)
    for i in range(bcnt):
        encode_block(stream, data[i * block_size : (i + 1) * block_size])
    if bcnt * block_size < len(data):
        encode_block(stream, data[bcnt * block_size :])
    stream.flush()
    result = len(stream.byte_stream)
    stream.write(file_path)
    return result


# df = pd.read_csv("test.csv")
# print(
#     ts2diff_encode(
#         df["value"].tolist(),
#         "result.bin",
#     )
# )
# def ts2diff_com(data: list[int]) -> float:
#     return ts2diff(data) / (len(data) * 32)
