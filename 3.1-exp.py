import os
import pandas as pd
from algorithm import *
from typing import Callable
import time

Encoder = Callable[[list[int], str], int]
EncoderParam = Callable[[list[int], str], tuple[int, float]]
Decoder = Callable[[str], list[int]]


def exp(
    data: list[int],
    encode: Encoder,
    decode: Decoder,
) -> tuple[int, float, float]:
    result_path = "result.bin"
    start = time.time()
    result = encode(data, result_path)
    end = time.time()
    encoding_time = (end - start) / len(data)
    start = time.time()
    data_new = decode(result_path)
    end = time.time()
    decoding_time = (end - start) / len(data)
    if data != data_new:
        raise ValueError("Decode fails.")
    return (result / (len(data) * 4), encoding_time, decoding_time)


def exp_param(
    data: list[int],
    encode: EncoderParam,
    decode: Decoder,
) -> tuple[int, float, float]:
    result_path = "result.bin"
    result, encoding_time = encode(data, result_path)
    encoding_time /= len(data)
    start = time.time()
    data_new = decode(result_path)
    end = time.time()
    decoding_time = (end - start) / len(data)
    if data != data_new:
        raise ValueError("Decode fails.")
    return (result / (len(data) * 4), encoding_time, decoding_time)


if not os.path.exists("exp_result"):
    os.makedirs("exp_result")

algorithms: list[tuple[str, Encoder, Decoder]] = [
    ("gorilla", gorilla_encode, gorilla_decode),
    ("rle", rle_encode, rle_decode),
    ("plain", plain_encode, plain_decode),
    ("ts_2diff", ts2diff_encode, ts_2diff_decode),
]

algorithmsParam: list[tuple[str, EncoderParam, Decoder]] = [
    ("period", period_encode_param, period_decode),
]

exp_data = {
    "dataset": [],
    "name": [],
    "algorithm": [],
    "compress_ratio": [],
    "encoding_time": [],
    "decoding_time": [],
}

files = os.listdir("data")
for file in files:
    dataset = file.split("_")[0]
    file_pull = os.path.join("data", file)
    df = pd.read_csv(file_pull)
    data = df["value"].tolist()
    for algorithm, encode, decode in algorithms:
        (result, encoding_time, decoding_time) = exp(data, encode, decode)
        exp_data["dataset"].append(dataset)
        exp_data["name"].append(file)
        exp_data["algorithm"].append(algorithm)
        exp_data["compress_ratio"].append(result)
        exp_data["encoding_time"].append(encoding_time)
        exp_data["decoding_time"].append(decoding_time)
        print(dataset, file, algorithm, result, encoding_time, decoding_time)

    for algorithm, encode, decode in algorithmsParam:
        (result, encoding_time, decoding_time) = exp_param(data, encode, decode)
        exp_data["dataset"].append(dataset)
        exp_data["name"].append(file)
        exp_data["algorithm"].append(algorithm)
        exp_data["compress_ratio"].append(result)
        exp_data["encoding_time"].append(encoding_time)
        exp_data["decoding_time"].append(decoding_time)
        print(dataset, file, algorithm, result, encoding_time, decoding_time)

exp_data_df = pd.DataFrame(data=exp_data)
exp_data_df.to_csv(os.path.join("exp_result", "exp.csv"), index=False)
