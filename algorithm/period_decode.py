import numpy as np
from mylib.byteToys import byteInToys
from mylib.round import comp_round_inverse

max_size: int = 2**32 - 1
max_value: int = 2**32 - 1
group_size = 8


def descending_bit_packing_decode(stream: byteInToys) -> list[int]:
    sgn = stream.decode(1)
    data_length = stream.decode(max_size.bit_length())
    data = [0] * data_length
    index_length = stream.decode(max_size.bit_length())
    index = [0] * index_length
    if index_length == 0:
        return [0] * data_length
    for i in range(0, index_length, group_size):
        max_len = stream.decode(max_size.bit_length().bit_length())
        for j in range(i, min(i + group_size, index_length)):
            index[j] = stream.decode(max_len)
    first_len = stream.decode(max_value.bit_length().bit_length())
    current_len = first_len
    for i in range(index_length):
        if sgn:
            current_sgn = stream.decode(1)
        data[index[i]] = stream.decode(current_len)
        current_len = data[index[i]].bit_length()
        if sgn and current_sgn == 1:
            data[index[i]] *= -1
    return data


def separate_storage_decode(stream: byteInToys) -> list[int]:
    data_length = stream.decode(max_size.bit_length())
    sgn = [0] * data_length
    data = [0] * data_length
    D = stream.decode(max_value.bit_length().bit_length())
    for i in range(data_length):
        sgn[i] = stream.decode(1)
        data[i] = stream.decode(D)
    high = descending_bit_packing_decode(stream)
    for i in range(data_length):
        data[i] = (1 if sgn[i] == 0 else -1) * (high[i] << D | data[i])
    return data


def period_decode(file_path: str) -> list[int]:
    stream = byteInToys(file_path)
    p = stream.decode(max_size.bit_length())
    if p == 0:
        return separate_storage_decode(stream)
    beta_sgn = stream.decode(1)
    beta = stream.decode(max_value.bit_length().bit_length())
    if beta_sgn == 1:
        beta *= -1
    dataf = comp_round_inverse(descending_bit_packing_decode(stream), beta)
    res = np.cumsum(separate_storage_decode(stream)).tolist()
    k = (len(res) + p - 1) // p
    return (
        np.tile(
            np.round(np.fft.irfft(dataf, n=p) / k).astype(np.int64),
            reps=(k,),
        )[: len(res)]
        - res
    ).tolist()


# print(period_decode("result.bin"))
