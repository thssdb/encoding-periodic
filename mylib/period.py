# 计算自相关函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml


# 计算多项式乘法
def poly_mul(p1, p2):
    # 创建两个多项式
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # 计算它们的乘积
    n = 2 ** int(np.ceil(np.log2(len(p1) + len(p2) - 1)))
    p1 = np.pad(p1, (0, n - len(p1)))
    p2 = np.pad(p2, (0, n - len(p2)))
    result = np.fft.ifft(np.fft.fft(p1) * np.fft.fft(p2)).real
    return result


with open(os.path.join("mylib", "pure_draw_config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.full_load(f)


# 归一化
def normalize(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_normalized = (x - x_mean) / x_std
    return x_normalized


# 计算自相关函数
def self_corr(x):
    x = normalize(x)
    N = len(x)
    result = []
    for i in range(int(N * (2 / 3))):
        sum = 0
        for j in range(N - i):
            sum += x[j] * x[j + i]
        sum /= N - i
        result.append(sum)
    return result


# 用fft加速计算自相关函数
def self_corr_fast(x):
    x = normalize(x)
    N = len(x)
    x_self = poly_mul(x, np.flip(x))
    result = []
    for i in range(int(N * (2 / 3))):
        sum = x_self[N - 1 - i]
        sum /= N - i
        result.append(sum)
    return result


# 给出数组中是邻近k个内最大的数的下标数组
def pink_local_max(data, k=config["k"]):
    result = []
    for i in range(k, len(data) - k):
        max_val = data[i]
        for j in range(max(0, i - k), min(len(data), i + k)):
            max_val = max(max_val, data[j])
        if max_val == data[i]:
            result.append(i)
    return result


# # 求出周期长度
# def get_period(data, p=config["p"]):
#     data_corr = self_corr(data)
#     points = pink_local_max(data_corr)
#     for point in points:
#         if point > 0 and data_corr[point] > p:
#             return point
#     return 0


# 借助fft求出周期长度
def get_period(data, p=config["p"]):
    data_corr = self_corr_fast(data)
    points = pink_local_max(data_corr)
    for point in points:
        if point > 0 and data_corr[point] > p:
            return point
    return 0


# test_length = 1000

# data = pd.read_csv(os.path.join("raw", "{}.csv".format(config["filename"])))
# data = data["value"][:test_length]
# data_corr = self_corr(data)
# p = get_period(data)
# plt.plot(data)
# plt.savefig(os.path.join("pure_fig", "data{}-origin.png".format(test_length)))
# plt.clf()
# plt.plot(data_corr)
# plt.scatter([p], [data_corr[p]], color="red")
# plt.savefig(os.path.join("pure_fig", "data{}-corr.png".format(test_length)))
# plt.clf()
