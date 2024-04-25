import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from mylib.period import get_period

FONT_SIZE = 10

if not os.path.exists("result"):
    os.makedirs("result")

with open("toys.yml", "r", encoding="utf-8") as f:
    toys = yaml.full_load(f)

file = toys["period-complementation"]

data = pd.read_csv(os.path.join("data", f"{file['name']}.csv"))
data = data["value"].tolist()
p = get_period(data)
# print(p)

fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0, 0].plot(data[: int(p * file["not_full_period"])])
axs[0, 0].set_title("(a) Data with incomplete periods", fontsize=FONT_SIZE)

dataf_not_full = np.abs(np.fft.fft(data[: int(p * file["not_full_period"])]))[
    : file["frequency_length"]
]

for i in range(len(dataf_not_full)):
    axs[0, 1].vlines(x=i, ymin=0, ymax=dataf_not_full[i], linewidth=1, color="brown")

index = list(range(len(dataf_not_full)))
index.sort(key=lambda x: dataf_not_full[x], reverse=True)
k = 6
for i in range(k):
    axs[0, 1].scatter(
        x=index[i],
        y=dataf_not_full[index[i]],
        marker="o",
        facecolors="none",
        edgecolors="brown",
    )
axs[0, 1].axhline(
    y=2 ** (int(np.log2(dataf_not_full[index[k - 1]])) - 1), color="red", linestyle="-"
)
# axs[0, 1].plot(
#     dataf_not_full,
#     color="orange",
# )
axs[0, 1].set_ylim(bottom=0)
axs[0, 1].set_title("(b) Frequency data with incomplete periods", fontsize=FONT_SIZE)

axs[1, 0].plot(data[: p * file["full_period"]])
axs[1, 0].set_title("(c) Data with complete periods", fontsize=FONT_SIZE)

dataf_full = np.abs(np.fft.fft(data[: p * file["full_period"]]))[
    : file["frequency_length"]
]

# axs[1, 1].plot(
#     dataf_full,
#     color="orange",
# )

for i in range(len(dataf_full)):
    axs[1, 1].vlines(x=i, ymin=0, ymax=dataf_full[i], linewidth=1, color="brown")

# axs[1, 1].scatter(
#     list(range(0, file["frequency_length"], file["full_period"])),
#     dataf_full[:: file["full_period"]],
#     color="orange",
# )


index = list(range(len(dataf_full)))
index.sort(key=lambda x: dataf_full[x], reverse=True)
for i in range(k):
    axs[1, 1].scatter(
        x=index[i],
        y=dataf_full[index[i]],
        marker="o",
        facecolors="none",
        edgecolors="brown",
    )
axs[1, 1].axhline(
    y=2 ** (int(np.log2(dataf_full[index[k - 1]])) - 1), color="red", linestyle="-"
)

axs[1, 1].set_ylim(bottom=0)
axs[1, 1].set_xticks(range(0, len(dataf_full), file["full_period"]))
axs[1, 1].set_title("(d) Frequency data with complete periods", fontsize=FONT_SIZE)

for i in range(2):
    for j in range(2):
        axs[i, j].ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useOffset=False
        )
plt.tight_layout()
plt.savefig(os.path.join("result", "period-complementation-example.png"))
plt.savefig(os.path.join("result", "period-complementation-example.eps"))
plt.clf()

FIGURE_SIZE = (3, 2)

plt.figure(figsize=FIGURE_SIZE)
plt.plot(dataf_not_full, color="orange")
# plt.xticks([])
# plt.yticks([])
plt.savefig(
    os.path.join("result", "period-complementation-dataf-not-full.png"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join("result", "period-complementation-dataf-not-full.eps"),
    bbox_inches="tight",
)
plt.clf()

plt.figure(figsize=FIGURE_SIZE)
plt.plot(dataf_full, color="orange")
plt.scatter(
    list(range(0, file["frequency_length"], file["full_period"])),
    dataf_full[:: file["full_period"]],
    color="orange",
)
# plt.xticks([])
# plt.yticks([])
plt.savefig(
    os.path.join("result", "period-complementation-dataf-full.png"), bbox_inches="tight"
)
plt.savefig(
    os.path.join("result", "period-complementation-dataf-full.eps"), bbox_inches="tight"
)
plt.clf()
