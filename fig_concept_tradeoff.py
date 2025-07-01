import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "lines.linewidth": 1.5,
    }
)

color_freq = "#8da0cb"
color_resid = "#fc8d62"
color_total = "#66c2a5"
color_opt_marker = "#f8cecc"

beta_conceptual = np.linspace(2, 18, 200)

cost_freq_conceptual = 1000 / (beta_conceptual - 1) + 20000 * np.exp(
    -(beta_conceptual - 2) / 4
)

cost_resid_conceptual = 80 * (beta_conceptual - 2) ** 2 + 500

cost_total_conceptual = cost_freq_conceptual + cost_resid_conceptual

min_idx_conceptual = np.argmin(cost_total_conceptual)
optimal_beta_conceptual = beta_conceptual[min_idx_conceptual]
min_cost_total_conceptual = cost_total_conceptual[min_idx_conceptual]

fig, ax = plt.subplots(figsize=(4.5, 2.5))

ax.plot(
    beta_conceptual,
    cost_freq_conceptual,
    "--",
    color=color_freq,
    label=r"Frequency Cost $L(\hat{F}_{\beta})$",
)
ax.plot(
    beta_conceptual,
    cost_resid_conceptual,
    ":",
    color=color_resid,
    label=r"Residual Cost $L(R_{\beta})$",
)
ax.plot(
    beta_conceptual,
    cost_total_conceptual,
    "-",
    color=color_total,
    linewidth=2.5,
    label="Total Cost",
)

ax.axvline(optimal_beta_conceptual, color="grey", linestyle="--", lw=1.0, zorder=0)

ax.plot(
    optimal_beta_conceptual,
    min_cost_total_conceptual,
    "*",
    color=color_opt_marker,
    markersize=16,
    markeredgecolor="black",
    markeredgewidth=0.5,
    label=r"Optimal Trade-off ($\beta^*$)",
    zorder=10,
)

ax.set_xlabel(r"Quantization Parameter ($\beta$)")
ax.set_ylabel("Theoretical Encoding Cost")

ax.set_xticks([])
ax.set_yticks([])

ax.legend(frameon=True, framealpha=0.75, fancybox=True, loc="upper center")

ax.set_xlim(beta_conceptual.min(), beta_conceptual.max())
ax.set_ylim(0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout(pad=0.2)

from config import FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "flea_conceptual_tradeoff.pdf", format="pdf", bbox_inches="tight"
)
plt.savefig(
    FIGURES_DIR / "flea_conceptual_tradeoff.eps", format="eps", bbox_inches="tight"
)
print("Conceptual trade-off plot saved.")

plt.show()
