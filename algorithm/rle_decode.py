from mylib.byteToys import byteInToys


def rle_decode(file_path: str) -> list[int]:
    stream = byteInToys(file_path)
    result = []
    pairs_length = stream.decode(32)
    for tmp in range(pairs_length):
        value = stream.decode(32)
        type = stream.decode(1)
        if type == 0:
            result.append(value)
        else:
            repeat = stream.decode(6)
            for tmp2 in range(repeat):
                result.append(value)
    for i in range(len(result)):
        if result[i] >= 1 << 31:
            result[i] -= 1 << 32
    return result


# print(rle_decode("result.bin"))
