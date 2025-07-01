import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_DIR, DATA_NO_PERIOD_DIR, FIGURES_DIR
from algorithm.frequency_encoding_v2 import (
    get_variable_length_decreasing_encoding_prefix_length,
    get_descending_bit_packing_suffix_length,
)
from algorithm.variable_length_decreasing_encoding import get_suffix_length


def plot_split_point_tradeoff(prefix_costs, suffix_costs, output_dir):
    """
    Creates and saves a professional plot showing the trade-off for the split point p.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "legend.fontsize": 9,
        }
    )

    total_costs = prefix_costs + suffix_costs
    split_points_x = np.arange(len(total_costs))

    min_idx = np.argmin(total_costs)
    optimal_p = split_points_x[min_idx]
    min_cost = total_costs[min_idx]

    color_gvle = "#8da0cb"
    color_dbp = "#fc8d62"
    color_total = "#66c2a5"
    color_opt_marker = "#f8cecc"

    fig, ax = plt.subplots(figsize=(6, 6 * 2 / 3))

    ax.plot(
        split_points_x, prefix_costs, "-", color=color_gvle, label=r"GVLE Cost (Prefix)"
    )
    ax.plot(
        split_points_x, suffix_costs, "-", color=color_dbp, label=r"DBP Cost (Suffix)"
    )
    ax.plot(
        split_points_x,
        total_costs,
        "-",
        color=color_total,
        linewidth=2,
        label="Total Frequency Cost",
    )

    ax.plot(
        optimal_p,
        min_cost,
        "*",
        color=color_opt_marker,
        markersize=16,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=f"Optimal Point ($p^*={optimal_p}$)",
        zorder=10,
    )

    ax.axvline(optimal_p, color="grey", linestyle="--", lw=1.0, zorder=0)

    ax.set_xlabel(r"Split Point ($p$)")
    ax.set_ylabel("Encoding Cost (bits)")
    ax.legend()

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0, len(total_costs))

    fig.tight_layout()
    output_filename = output_dir / "p_split_tradeoff"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.eps", format="eps")

    print(f"Split point trade-off plot saved to {output_filename}.pdf/.eps")


if __name__ == "__main__":
    file = "stock_AAPL_2006-01-01_to_2018-01-01.csv"
    df = pd.read_csv(DATA_NO_PERIOD_DIR / file)
    data = df["value"].to_numpy(dtype="int64")
    beta = 16
    frequency = np.fft.rfft(data)
    frequency_imag = np.imag(frequency)
    frequency_imag_quantized = np.round(frequency_imag / (2**beta)).astype(np.int64)

    suffix_length, _ = get_suffix_length(frequency_imag_quantized)
    variable_length_decreasing_encoding_prefix_length = (
        get_variable_length_decreasing_encoding_prefix_length(
            frequency_imag_quantized, suffix_length
        )
    )
    descending_bit_packing_suffix_length = get_descending_bit_packing_suffix_length(
        frequency_imag_quantized
    )

    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    plot_split_point_tradeoff(
        variable_length_decreasing_encoding_prefix_length,
        descending_bit_packing_suffix_length,
        FIGURES_DIR,
    )
