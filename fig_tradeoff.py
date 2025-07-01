import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from typing import List
from config import DATA_DIR, FIGURES_DIR
from encoder.flea_ablation import FLEAEncoder

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "gray",
        "lines.linewidth": 1.5,
    }
)

color_freq = "#8da0cb"
color_resid = "#fc8d62"
color_total = "#66c2a5"
color_opt = "#f8cecc"

beta_values = np.arange(0, 20, 2)
df = pd.read_csv(DATA_DIR / "volt_datanormal150km_vdata2B1.csv")
data = df["value"].to_numpy(dtype="int64")
results: List[pd.DataFrame] = []
for beta in range(0, 20, 2):
    encoder = FLEAEncoder(is_given_beta=True, given_beta=beta)
    result = encoder.exp(data)
    results.append(result)
result = pd.concat(results, ignore_index=True)

cost_freq_values = result["frequency_encoding_length"].to_numpy(dtype="int64")
cost_resid_values = result["residual_encoding_length"].to_numpy(dtype="int64")
cost_total_values = cost_freq_values + cost_resid_values

min_idx = np.argmin(cost_total_values)
optimal_beta = beta_values[min_idx]
min_cost_total = cost_total_values[min_idx]

fig, ax = plt.subplots(figsize=(6, 6 * 2 / 3))

ax.plot(
    beta_values,
    cost_freq_values,
    "^--",
    color=color_freq,
    markersize=10,
    label=r"$L(\hat{\mathbf{F}}_{\beta})$",
)
ax.plot(
    beta_values,
    cost_resid_values,
    "v:",
    color=color_resid,
    markersize=10,
    label=r"$L(\mathbf{R}_{\beta})$",
)
ax.plot(
    beta_values,
    cost_total_values,
    "s-",
    color=color_total,
    linewidth=2.0,
    markersize=10,
    label="Total Cost",
)

ax.axvline(optimal_beta, color="grey", linestyle="--", lw=1.0, zorder=0)

ax.plot(
    optimal_beta,
    min_cost_total,
    "*",
    color=color_opt,
    markersize=16,
    markeredgecolor="black",
    markeredgewidth=0.5,
    label=f"Optimal (β*={optimal_beta})",
    zorder=10,
)

ax.set_xlabel(r"Quantization Parameter ($\beta$)")
ax.set_ylabel("Estimated Cost (bits)")

ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

ax.legend(frameon=True, fancybox=True)

ax.set_xlim(beta_values.min() - 1, beta_values.max() + 1)
ax.set_ylim(0, cost_total_values.max() * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout(pad=0.2)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "flea_tradeoff.pdf", format="pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "flea_tradeoff.eps", format="eps", bbox_inches="tight")
