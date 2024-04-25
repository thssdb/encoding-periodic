from mylib.byteToys import byteInToys


def gorilla_decode(file_path: str) -> list[int]:
    stream = byteInToys(file_path)
    first_value = stream.decode(32)
    length = stream.decode(32)
    result = [first_value]
    l = t = -1
    for i in range(1, length):
        type = stream.decode(1)
        if type == 0:
            result.append(result[-1])
        else:
            type = stream.decode(1)
            if type == 0:
                x = stream.decode(32 - l - t)
                result.append(result[-1] ^ (x << t))
            else:
                l = stream.decode(6)
                t = stream.decode(6)
                x = stream.decode(32 - l - t)
                result.append(result[-1] ^ (x << t))

    for i in range(length):
        if result[i] >= 1 << 31:
            result[i] -= 1 << 32
    return result


# print(gorilla_decode("result.bin"))
