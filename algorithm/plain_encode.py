import numpy as np
import pandas as pd
from mylib.byteToys import byteOutToys


def plain_encode(data: list[int], file_path: str):
    stream = byteOutToys()
    stream.encode(len(data), 32)
    for x in data:
        stream.encode(x, 32)
    stream.flush()
    result = len(stream.byte_stream)
    stream.write(file_path)
    return result


# df = pd.read_csv("test.csv")
# print(
#     plain_encode(
#         df["value"].tolist(),
#         "result.bin",
#     )
# )
# def ts2diff_com(data: list[int]) -> float:
#     return ts2diff(data) / (len(data) * 32)
