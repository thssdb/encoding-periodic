import pandas as pd
from mylib.byteToys import byteOutToys

max_repeat = 63


def rle_encode(data: list[int], file_path: str) -> int:
    stream = byteOutToys()
    pairs = []
    last_value = data[0]
    last_length = 1
    for i in range(1, len(data)):
        if last_length == max_repeat or data[i] != last_value:
            pairs.append((last_value, last_length))
            last_value = data[i]
            last_length = 1
        else:
            last_length += 1
    pairs.append((last_value, last_length))
    stream.encode(len(pairs), 32)
    for value, repeat in pairs:
        stream.encode(value, 32)
        if repeat == 1:
            stream.encode(0, 1)
        else:
            stream.encode(1, 1)
            stream.encode(repeat, 6)
    stream.flush()
    result = len(stream.byte_stream)
    stream.write(file_path)
    return result


# df = pd.read_csv("test.csv")
# print(rle_encode(df["value"].tolist(), "result.bin"))
