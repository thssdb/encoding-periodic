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


def plot_frequency_component(ax, data, ylabel):
    """A generalized function to plot one frequency component (real or imag)."""
    n = len(data)
    indices = np.arange(n)

    markerline, stemlines, baseline = ax.stem(
        indices, data, linefmt="grey", markerfmt="o", basefmt="k-"
    )
    plt.setp(markerline, "markersize", 3, "color", "#9467bd")
    plt.setp(stemlines, "linewidth", 0.8)
    plt.setp(baseline, "linewidth", 1)

    for beta in range(14, 19, 2):
        ax.axhline(y=2**beta, color="red", linestyle="--", linewidth=0.5)
        ax.text(
            n - 1,
            2**beta + 0.5,
            r"$2^\beta, \beta={}$".format(beta),
            color="red",
            fontsize=14,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel(r"Frequency Coefficient Index")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, n)
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":

    N_COEFFS_TO_PLOT = 120

    file = DATA_NO_PERIOD_DIR / "stock_AAPL_2006-01-01_to_2018-01-01.csv"
    real_coeffs, imag_coeffs = load_and_process_data(file, np.int64(3))

    if real_coeffs is not None:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))

        plot_frequency_component(
            ax=axes,
            data=imag_coeffs[:N_COEFFS_TO_PLOT],
            ylabel=r"Magnitude",
        )

        plt.tight_layout(pad=1.0)

        output_filename = FIGURES_DIR / "different_beta"
        plt.savefig(f"{output_filename}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{output_filename}.eps", format="eps", bbox_inches="tight")

        print(f"Figure saved to {output_filename}.pdf and .eps")
