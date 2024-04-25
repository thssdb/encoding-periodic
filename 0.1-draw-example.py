import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from mylib.period import get_period
from mylib.round import comp_round

if not os.path.exists("result"):
    os.makedirs("result")

with open("toys.yml", "r", encoding="utf-8") as f:
    toys = yaml.full_load(f)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))

file = toys["old-method"]

data = pd.read_csv(os.path.join("data", f"{file['name']}.csv"))
data = data["value"].tolist()
p = get_period(data)
data = data[: int(file["full_period"] * p)]

data_a = data
axs[0, 0].plot(data_a)
axs[0, 0].set_ylim(top=17000)
axs[0, 0].set_title("(a) Origin data")

dataf = np.fft.fft(data)
data_b = np.abs(dataf)[: file["frequency_length"]]
# print(data_b)
for i in range(len(data_b)):
    axs[0, 1].vlines(x=i, ymin=0, ymax=data_b[i], linewidth=1, color="brown")
    axs[0, 1].scatter(
        x=i,
        y=data_b[i],
        marker="o",
        facecolors="none",
        edgecolors="brown",
    )
# axs[0, 1].axhline(y=2**15, color="red", linestyle="-")
axs[0, 1].set_ylim(bottom=0)
axs[0, 1].set_title("(b) Frequency data")

lossy_dataf, _ = comp_round(dataf, 15)
lossy_data = np.fft.ifft(lossy_dataf)

data_c = lossy_data
axs[1, 0].plot(data_c)
axs[1, 0].set_ylim(top=17000)
axs[1, 0].set_title("(c) Lossy data")

data_d = data - lossy_data
axs[1, 1].plot(data_d)
axs[1, 1].set_title("(d) Residual data")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join("exp_result", "example.eps"))
plt.savefig(os.path.join("exp_result", "example.png"))
