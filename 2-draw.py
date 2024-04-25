import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from mylib.period import get_period
from mylib.round import comp_round

if not os.path.exists("result"):
    os.makedirs("result")


def draw(data, figname: str, color: str = None):
    FIGURE_SIZE = (2.5, 2)
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(data, color=color)
    plt.yticks([])
    plt.savefig(os.path.join("result", figname), bbox_inches="tight")
    plt.clf()


with open("toys.yml", "r", encoding="utf-8") as f:
    toys = yaml.full_load(f)

file = toys["algorithm"]

data = pd.read_csv(os.path.join("data", f"{file['name']}.csv"))
data = data["value"].tolist()
p = get_period(data)

data_a = data[: int(p * file["not_full_period"])]
draw(data_a, "data_a.svg")

data_b = data[: p * file["full_period"]]
draw(data_b, "data_b.svg")

dataf = np.fft.fft(data_b)

data_c = np.abs(dataf)[: file["frequency_length"]]
# frequency_draw(data_c, "data_c.svg", color="brown")
for i in range(len(data_c)):
    plt.vlines(x=i, ymin=0, ymax=data_c[i], linewidth=1, color="brown")
    plt.scatter(
        x=i,
        y=data_c[i],
        marker="o",
        facecolors="none",
        edgecolors="brown",
    )
plt.ylim(bottom=0)
plt.yticks([])
plt.savefig(os.path.join("result", "data_c.svg"), bbox_inches="tight")
plt.clf()

for i in range(len(dataf)):
    if i % file["full_period"] != 0:
        dataf[i] = 0
dataf, tmp = comp_round(dataf, 10)

data_d = np.abs(dataf)[: file["frequency_length"]]
# frequency_draw(data_d, "data_d.svg", color="brown")
for i in range(0, len(data_d), file["full_period"]):
    plt.vlines(x=i, ymin=0, ymax=data_d[i], linewidth=1, color="brown")
    plt.scatter(
        x=i,
        y=data_d[i],
        marker="o",
        facecolors="none",
        edgecolors="brown",
    )
plt.ylim(bottom=0)
plt.yticks([])
plt.savefig(os.path.join("result", "data_d.svg"), bbox_inches="tight")
plt.clf()

data_e = np.abs(np.fft.ifft(dataf))
draw(data_e, "data_e.svg")

data_f = data_a - data_e[: int(p * file["not_full_period"])]
draw(data_f, "data_f.svg")

data_g = np.diff(data_f)
draw(data_g, "data_g.svg")
