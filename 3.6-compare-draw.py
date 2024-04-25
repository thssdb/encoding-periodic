import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_PATH = os.path.join("exp_result", "exp_compare_draw.csv")
RESULT_PATH = os.path.join("exp_result", "compare.png")
RESULT_PATH_EPS = os.path.join("exp_result", "compare.eps")

df = pd.read_csv(INPUT_PATH)
type_array = ["previous", "average"]
dataset_array = [
    "temperature",
    "volt",
    "dianwang",
    "guoshou",
    "liantong",
    "yinlian",
]
# print(type_array)

bar_width = 0.8 / len(type_array)
index_offset = np.arange(len(dataset_array))

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for idx, type in enumerate(type_array):
    subset = df[df["algorithm"] == type]
    bar_x = index_offset + bar_width * idx  # 计算每个柱子的中心点x坐标
    ax.bar(
        bar_x,
        subset["compress_ratio"],
        width=bar_width,
        label=type,
    )

# ax.set_title("(a) MSE loss of different method")
ax.set_xlabel("Database")
ax.set_ylabel("Compression Ratio")
# ax.set_yscale("log")

ax.set_xticks(
    index_offset + bar_width * (len(type_array) - 1) / 2
)  # 设置x轴刻度标签的位置
ax.set_xticklabels(dataset_array, rotation=0)  # 设置x轴标签

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1),
    ncol=len(type_array),
)  # 显示图例


# for idx, type in enumerate(type_array):
#     subset = df[df["type"] == type]
#     bar_x = index_offset + bar_width * idx  # 计算每个柱子的中心点x坐标
#     axs[1].bar(
#         bar_x,
#         subset["time"],
#         width=bar_width,
#         label=type,
#     )

# axs[1].set_title("(b) Time cost of different method")
# axs[1].set_xlabel("Database")
# axs[1].set_ylabel("Time cost(s)")
# axs[1].set_yscale("log")

# axs[1].set_xticks(
#     index_offset + bar_width * (len(type_array) - 1) / 2
# )  # 设置x轴刻度标签的位置
# axs[1].set_xticklabels(dataset_array, rotation=0)  # 设置x轴标签

plt.tight_layout()
plt.subplots_adjust(top=0.9)
# plt.show()
plt.savefig(RESULT_PATH)
plt.savefig(RESULT_PATH_EPS)
