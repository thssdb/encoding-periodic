import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def draw(name: str, column_name: str, log_scale=False):
    df = pd.read_csv(os.path.join("exp_result", f"{name}.csv"))
    custom_order = {
        "temperature": 0,
        "volt": 1,
        "dianwang": 2,
        "guoshou": 3,
        "liantong": 4,
        "yinlian": 5,
    }
    df["order"] = df["dataset"].map(custom_order)
    df = df.sort_values(by="order").reset_index()
    del df["order"]
    compression_algorithms = ["period", "ts_2diff", "gorilla", "rle"]

    show_name = {
        "period": "PERIOD",
        "ts_2diff": "TS_2DIFF",
        "gorilla": "GORILLA",
        "rle": "RLE",
        "plain": "PLAIN",
    }
    df_melted = pd.melt(
        df,
        id_vars="dataset",
        var_name="Compression Algorithm",
        value_name=column_name,
    )

    bar_width = 0.8 / len(compression_algorithms)  # 柱子宽度
    index_offset = np.arange(len(df["dataset"]))  # 获取数据库索引

    fig, ax = plt.subplots(figsize=(5.5, 5.5 * 0.5))

    for idx, algorithm in enumerate(compression_algorithms):
        subset = df_melted[df_melted["Compression Algorithm"] == algorithm]
        bar_x = index_offset + bar_width * idx  # 计算每个柱子的中心点x坐标
        ax.bar(
            bar_x,
            subset[column_name],
            width=bar_width,
            label=show_name[algorithm],
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(column_name)
    # ax.set_title("Compression Performance Across Databases")
    ax.set_xticks(
        index_offset + bar_width * (len(compression_algorithms) - 1) / 2
    )  # 设置x轴刻度标签的位置
    ax.set_xticklabels(df["dataset"].unique(), rotation=0)  # 设置x轴标签
    ax.legend(
        # title="Compression Algorithms",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,
    )  # 显示图例

    plt.tight_layout()
    plt.savefig(os.path.join("exp_result", f"{name}.png"))
    plt.savefig(os.path.join("exp_result", f"{name}.eps"))
    plt.clf()
    # plt.savefig("tmp/fig/result0329.png")


draw("compress_ratio", "Compression Ratio")
draw("encoding_time", "Average Encoding Time Per Point(s)", log_scale=True)
draw("decoding_time", "Average Decoding Time Per Point(s)", log_scale=True)
