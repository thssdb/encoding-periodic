import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_DIR, DATA_NO_PERIOD_DIR, FIGURES_DIR
from pathlib import Path

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
    }
)


def load_and_process_data(file_path: Path, beta: np.int64):
    df = pd.read_csv(file_path)

    data = df["value"].to_numpy(dtype="int64")
    frequency = np.fft.fft(data)

    freq_quantized_real = np.abs(
        np.round(np.real(frequency) / (2**beta)).astype(np.int64)
    )
    freq_quantized_imag = np.abs(
        np.round(np.imag(frequency) / (2**beta)).astype(np.int64)
    )

    return freq_quantized_real, freq_quantized_imag


def plot_frequency_component(ax, data, ylabel, split_point):
    """A generalized function to plot one frequency component (real or imag)."""
    n = len(data)
    indices = np.arange(n)

    markerline, stemlines, baseline = ax.stem(
        indices, data, linefmt="grey", markerfmt="o", basefmt="k-"
    )
    plt.setp(markerline, "markersize", 3, "color", "#9467bd")
    plt.setp(stemlines, "linewidth", 0.8)
    plt.setp(baseline, "linewidth", 1)

    ax.axvspan(-0.5, split_point - 0.5, facecolor="#e5f3e4")
    ax.axvspan(split_point - 0.5, n - 0.5, facecolor="#ffecd9")

    text_y_pos = np.max(data) * 0.85
    ax.text(
        split_point / 2,
        text_y_pos,
        "Skewed Low-frequency\nCoefficients",
        ha="center",
        va="top",
        fontsize=14,
        color="#3C8D40",
    )
    ax.text(
        split_point + (n - split_point) / 2,
        text_y_pos,
        "Sparse High-frequency\nCoefficients",
        ha="center",
        va="top",
        fontsize=14,
        color="#C56300",
    )

    ax.axvline(x=split_point - 0.5, color="#e41a1c", linestyle="--", linewidth=2)
    ax.text(
        split_point + 3,
        np.max(data) * 0.9,
        "p",
        ha="center",
        va="bottom",
        fontsize=18,
        color="#e41a1c",
    )

    ax.set_xlabel(r"Frequency Coefficient Index")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, n)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":

    BETA = 18
    N_COEFFS_TO_PLOT = 120

    SPLIT_POINT_REAL = 28
    SPLIT_POINT_IMAG = 56

    file = DATA_NO_PERIOD_DIR / "stock_AAPL_2006-01-01_to_2018-01-01.csv"
    real_coeffs, imag_coeffs = load_and_process_data(file, np.int64(BETA))

    if real_coeffs is not None:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))

        plot_frequency_component(
            ax=axes,
            data=imag_coeffs[:N_COEFFS_TO_PLOT],
            ylabel=r"Magnitude",
            split_point=SPLIT_POINT_IMAG,
        )

        plt.tight_layout(pad=1.0)

        output_filename = FIGURES_DIR / "freq_data_regions"
        plt.savefig(f"{output_filename}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{output_filename}.eps", format="eps", bbox_inches="tight")

        print(f"Figure saved to {output_filename}.pdf and .eps")
