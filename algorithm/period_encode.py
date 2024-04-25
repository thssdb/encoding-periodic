import numpy as np
import pandas as pd
from mylib.byteToys import byteOutToys
from mylib.period import get_period
from mylib.round import comp_round
import time

max_size: int = 2**32 - 1
max_value: int = 2**32 - 1
group_size = 8


def get_cnt(data: list[int]) -> list[int]:
    cnt = [0] * (max_value.bit_length() + 1)
    for i in range(len(data)):
        if data[i] != 0:
            cnt[int(np.abs(data[i])).bit_length()] += 1
    return cnt


def bit_length_order(data: list[int]) -> list[int]:
    cnt = get_cnt(data)
    cnt[0] = 0
    for i in range(len(cnt) - 2, -1, -1):
        cnt[i] += cnt[i + 1]
    n = cnt[0]
    result = [0] * n
    for i in range(len(data) - 1, -1, -1):
        if data[i] != 0:
            result[cnt[int(np.abs(data[i])).bit_length()] - 1] = i
            cnt[int(np.abs(data[i])).bit_length()] -= 1
    return result


def descending_bit_packing(stream: byteOutToys, data: list[int], sgn=False):
    stream.encode(1 if sgn else 0, 1)
    index = bit_length_order(data)
    stream.encode(len(data), max_size.bit_length())
    stream.encode(len(index), max_size.bit_length())
    if len(index) == 0:
        return
    for i in range(0, len(index), group_size):
        max_len = 0
        for j in range(i, min(i + group_size, len(index))):
            max_len = max(max_len, index[j].bit_length())
        stream.encode(max_len, max_size.bit_length().bit_length())
        for j in range(i, min(i + group_size, len(index))):
            stream.encode(index[j], max_len)
    first_len = int(np.abs(data[index[0]])).bit_length()
    current_len = first_len
    stream.encode(first_len, max_value.bit_length().bit_length())
    for i in range(len(index)):
        if sgn:
            stream.encode(1 if data[index[i]] < 0 else 0, 1)
        stream.encode(int(np.abs(data[index[i]])), current_len)
        current_len = int(np.abs(data[index[i]])).bit_length()


def descending_bit_packing_estimate(cnt: list[int], n: int, sgn=False) -> int:
    sum1 = 0
    sum2 = 0
    for i in range(1, len(cnt)):
        sum1 += cnt[i]
        sum2 += cnt[i] * (i + (1 if sgn else 0))
    return n.bit_length() * sum1 + sum2


def calc_separate_storge_length(cnt: list[int], n: int, D: int) -> int:
    return n * (D + 1) + descending_bit_packing_estimate(cnt[D:], n)


# return: [result, D]
def separate_storage_estimate(data: list[int]) -> tuple[int, int]:
    cnt = get_cnt(data)
    result = calc_separate_storge_length(cnt, len(data), 0)
    D = 0
    for current_D in range(1, max_value.bit_length() + 1):
        tmp = calc_separate_storge_length(cnt, len(data), current_D)
        if tmp < result:
            result = tmp
            D = current_D
    return result, D


def separate_storage(stream: byteOutToys, data: list[int]):
    result, D = separate_storage_estimate(data)
    stream.encode(len(data), max_size.bit_length())
    stream.encode(D, max_value.bit_length().bit_length())
    # low-bit part
    for i in range(len(data)):
        # sgn bit
        stream.encode(1 if data[i] < 0 else 0, 1)
        stream.encode(int(np.abs(data[i])) & ((1 << D) - 1), D)
    # high-bit part
    descending_bit_packing(stream, (np.abs(data) // (1 << D)).tolist())


def get_res(data: list[int], dataf, p: int, k: int) -> list[int]:
    res: list[int] = (
        np.tile(
            np.round(np.fft.irfft(dataf, n=p) / k).astype(np.int64),
            reps=(k,),
        )[: len(data)]
        - data
    ).tolist()
    return res


def get_dataf_and_res(data: list[int], dataf, p: int, k: int, beta: int):
    dataf, ret = comp_round(dataf, beta)
    res = get_res(data, dataf, p, k)
    return dataf, ret, res


def encode_with_beta(
    stream: byteOutToys, data: list[int], dataf, p: int, k: int, beta: int
):
    dataf, ret, res = get_dataf_and_res(data, dataf, p, k, beta)
    descending_bit_packing(stream, ret, sgn=True)
    res = [res[0]] + np.diff(res).tolist()
    separate_storage(stream, res)


def encode_with_beta_estimate(data: list[int], dataf, p: int, k: int, beta: int) -> int:
    dataf, ret = comp_round(dataf, beta)
    max_len = 0
    for x in ret:
        max_len = max(max_len, x.bit_length())
    if max_len > max_value.bit_length():
        return -1
    result = descending_bit_packing_estimate(get_cnt(ret), len(ret), sgn=True)
    res = (
        np.tile(
            np.round(np.fft.irfft(dataf, n=p) / k).astype(np.int64),
            reps=(k,),
        )[: len(data)]
        - data
    ).tolist()
    res = [res[0]] + np.diff(res).tolist()
    tmp, D = separate_storage_estimate(res)
    result += tmp
    return result


def get_beta(data: list[int], dataf, p: int, k: int) -> int:
    result = encode_with_beta_estimate(data, dataf, p, k, 0)
    beta = 0
    for current_beta in range(-max_value.bit_length() // 2, max_value.bit_length() + 1):
        if current_beta != 0:
            tmp = encode_with_beta_estimate(data, dataf, p, k, current_beta)
            if result == -1 or (tmp != -1 and tmp < result):
                result = tmp
                beta = current_beta
    return beta


def period_encode(data: list[int], file_path: str, use_average=False) -> int:
    stream = byteOutToys()
    p = get_period(data)
    if p == 0:
        stream.encode(p, max_size.bit_length())
        separate_storage(stream, data)
    else:
        k = (len(data) + p - 1) // p
        if len(data) % p == 0:
            data_full = data
        else:
            if use_average == False:
                data_full = data + data[-p : -(len(data) % p)]
            else:
                data_extra = [0] * (p - len(data) % p)
                count = 0
                for i in range(0, len(data), p):
                    if i + p - 1 < len(data):
                        count += 1
                        for j in range(p - len(data) % p):
                            data_extra[len(data_extra) - 1 - j] += data[i + p - 1 - j]
                for i in range(len(data_extra)):
                    data_extra[i] = data_extra[i] // count
                data_full = data + data_extra
        dataf = np.fft.rfft(data_full)[::k]
        result = encode_with_beta_estimate(data, dataf, p, k, 0)
        beta = get_beta(data, dataf, p, k)
        stream.encode(p, max_size.bit_length())
        stream.encode(0 if beta >= 0 else 1, 1)
        stream.encode(np.abs(beta), max_value.bit_length().bit_length())
        encode_with_beta(stream, data, dataf, p, k, beta)
    stream.flush()
    result = len(stream.byte_stream)
    stream.write(file_path)
    return result


def period_encode_param(
    data: list[int], file_path: str, use_average=False
) -> tuple[int, float]:
    start = time.time()
    stream = byteOutToys()
    p = get_period(data)
    end = time.time()
    time_current = end - start
    if p == 0:
        start = time.time()
        stream.encode(p, max_size.bit_length())
        separate_storage(stream, data)
    else:
        k = (len(data) + p - 1) // p
        if len(data) % p == 0:
            data_full = data
        else:
            if use_average == False:
                data_full = data + data[-p : -(len(data) % p)]
            else:
                data_extra = [0] * (p - len(data) % p)
                count = 0
                for i in range(0, len(data), p):
                    if i + p - 1 < len(data):
                        count += 1
                        for j in range(p - len(data) % p):
                            data_extra[len(data_extra) - 1 - j] += data[i + p - 1 - j]
                for i in range(len(data_extra)):
                    data_extra[i] = data_extra[i] // count
                data_full = data + data_extra
        dataf = np.fft.rfft(data_full)[::k]
        result = encode_with_beta_estimate(data, dataf, p, k, 0)
        beta = get_beta(data, dataf, p, k)
        stream.encode(p, max_size.bit_length())
        stream.encode(0 if beta >= 0 else 1, 1)
        stream.encode(np.abs(beta), max_value.bit_length().bit_length())
        start = time.time()
        encode_with_beta(stream, data, dataf, p, k, beta)
    stream.flush()
    result = len(stream.byte_stream)
    stream.write(file_path)
    end = time.time()
    time_current += end - start
    return result, time_current


def period_result(data: list[int]) -> tuple[int, list[np.complex128], list[int]]:
    p = get_period(data)
    if p == 0:
        return 0, [], data
    else:
        k = (len(data) + p - 1) // p
        data_full = data + (data[-p : -(len(data) % p)] if len(data) % p != 0 else [])
        dataf = np.fft.rfft(data_full)[::k]
        beta = get_beta(data, dataf, p, k)
        dataf, ret, res = get_dataf_and_res(data, dataf, p, k, beta)
        return p, dataf.tolist(), res


def period_direct_result(data: list[int]) -> tuple[int, list[np.complex128], list[int]]:
    p = get_period(data)
    if p == 0:
        return 0, [], data
    else:
        k = (len(data) + p - 1) // p
        data_full = data + (data[-p : -(len(data) % p)] if len(data) % p != 0 else [])
        dataf = np.fft.rfft(data_full)[::k]
        res = get_res(data, dataf, p, k)
        return p, dataf.tolist(), res


# df = pd.read_csv("test.csv")
# print(period_encode(df["value"].tolist(), "result.bin"))
