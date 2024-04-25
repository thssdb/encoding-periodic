import numpy as np
import pandas as pd
from mylib.byteToys import byteOutToys


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


def gorilla_encode(data: list[int], file_path: str) -> int:
    stream = byteOutToys()
    lastl = lastt = -1
    # result = 32  # first_value
    stream.encode(data[0], 32)
    stream.encode(len(data), 32)
    for i in range(1, len(data)):
        x = data[i] ^ data[i - 1]
        if x == 0:
            # result += 1  # '0'
            stream.encode(0, 1)
        else:
            # result += 1  # '1'
            l = leadingzero(x)
            t = trailingzero(x)
            if lastl != -1 and l >= lastl and t >= lastt:
                # result += 1  # '0'
                # result += 32 - lastl - lastt
                stream.encode(2, 2)
                stream.encode(x >> lastt, 32 - lastl - lastt)
            else:
                # result += 1  # '1'
                # result += 6 + 6  # l,t
                # result += 32 - l - t
                stream.encode(3, 2)
                stream.encode(l, 6)
                stream.encode(t, 6)
                stream.encode(x >> t, 32 - l - t)
                lastl = l
                lastt = t
    stream.flush()
    result = len(stream.byte_stream)
    stream.write(file_path)
    return result


# def gorilla_com(data: list[int]) -> float:
#     return gorilla(data) / (len(data) * 32)


# df = pd.read_csv("test.csv")
# print(gorilla_encode(df["value"].tolist(), "result.bin"))
