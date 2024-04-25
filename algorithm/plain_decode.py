import numpy as np
from mylib.byteToys import byteInToys


def plain_decode(file_path: str) -> list[int]:
    stream = byteInToys(file_path)
    n = stream.decode(32)
    result = []
    for tmp in range(n):
        result.append(stream.decode(32))
    for i in range(len(result)):
        if result[i] >= 1 << 31:
            result[i] -= 1 << 32
    return result


# print(plain_decode("result.bin"))
