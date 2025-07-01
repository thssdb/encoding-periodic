import numpy as np
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from custom_tools.frequency import (
    time_to_frequency_quantized,
    frequency_to_time_quantized,
)

from config import DATA_DIR, FIGURES_DIR
from encoder.flea_ablation import FLEAEncoder


def plot_intro_four_panel_figure_preserved(time_series_data, output_dir):
    """
    Generates the 4-panel conceptual figure for the introduction, PRESERVING
    the original design of subplots (a) and (b).

    Args:
        time_series_data (np.ndarray): The 1D array of the time series to plot.
        output_dir (pathlib.Path): Directory to save the figure.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "svg.fonttype": "none",
        }
    )

    n = len(time_series_data)
    time_axis = np.arange(n)

    frequency_spectrum = np.fft.fft(time_series_data)
    frequency_count = int(n * 0.125)
    freq_axis = np.arange(frequency_count)
    freq_magnitudes = np.abs(frequency_spectrum)[:frequency_count]

    beta = 13

    magnitude_threshold = np.sqrt(2) * (2**beta)

    reconstructed_signal = frequency_to_time_quantized(
        time_to_frequency_quantized(time_series_data, beta), n, beta
    )

    residual_signal = time_series_data - reconstructed_signal.astype(np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5 * (3 / 4)))
    axes = axes.flatten()

    ax1 = axes[0]
    ax1.plot(time_axis, time_series_data, color="#377eb8", linewidth=1.0)
    ax1.set_title("(a) Periodic Time Series (Time Domain)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    local_window_start = n // 4
    local_window_width = n // 10
    ylim_ax1 = ax1.get_ylim()
    ax1.add_patch(
        Rectangle(
            (local_window_start, ylim_ax1[0]),
            local_window_width,
            ylim_ax1[1] - ylim_ax1[0],
            facecolor="grey",
            alpha=0.2,
            edgecolor="black",
            linestyle="--",
        )
    )
    ax1.text(
        local_window_start + local_window_width / 2,
        ylim_ax1[1] * 0.95,
        "Local View\n(e.g., Gorilla)",
        ha="center",
        va="top",
        fontsize=12,
        style="italic",
    )

    ax2 = axes[1]
    ax2.plot(freq_axis, freq_magnitudes, color="grey", alpha=0.5)
    pattern_mask = freq_magnitudes >= magnitude_threshold
    ax2.stem(
        freq_axis[pattern_mask],
        freq_magnitudes[pattern_mask],
        linefmt="#e41a1c",
        markerfmt="o",
        basefmt=" ",
        label="Structural Pattern",
    )
    noise_mask = ~pattern_mask
    ax2.stem(
        freq_axis[noise_mask][::5],
        freq_magnitudes[noise_mask][::5],
        linefmt="#9467bd",
        markerfmt=".",
        basefmt=" ",
        label="Noise & Details",
    )
    ax2.set_title("(b) Frequency Domain")
    ax2.set_xlabel("Frequency Index")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax2.yaxis.get_offset_text()
    offset_text.set_position((0, 1.02))
    offset_text.set_horizontalalignment("right")

    ax3 = axes[2]
    ax3.plot(time_axis, reconstructed_signal, color="#ff7f00", linewidth=1.0)
    ax3.set_title("(c) Lossy Reconstructed Series")
    ax3.set_xlabel("Time Step")
    ax3.set_ylim(ylim_ax1)
    ax3.set_ylabel("Value")

    ax4 = axes[3]
    ax4.scatter(
        time_axis,
        residual_signal,
        color="#2ca02c",
        s=10,
        alpha=0.5,
        edgecolor="none",
    )
    ax4.set_title("(d) Residual")
    ax4.set_xlabel("Time Step")
    ax4.set_ylim(ylim_ax1 - np.mean(ylim_ax1))
    ax4.set_ylabel("Value")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax in [ax2, ax4]:
            (ax.set_ylim(bottom=0) if ax is ax2 else None)

    fig.tight_layout(pad=0.0, h_pad=1.0)

    output_filename = output_dir / "introduction"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.svg", format="svg")
    print(f"4-panel conceptual figure saved to {output_filename}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "guoshou_value_from2020-01-02to2020-01-21_7500.csv")
    time_series_data = df["value"].to_numpy(dtype=np.int64)[:2048]

    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    plot_intro_four_panel_figure_preserved(
        time_series_data=time_series_data, output_dir=FIGURES_DIR
    )
