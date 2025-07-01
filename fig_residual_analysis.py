import os
from config import DATA_DIR, EXP_RESULTS_DIR
from encoder.gorilla import GorillaEncoder
from encoder.chimp import ChimpEncoder
from encoder.buff import BuffEncoder
from encoder.ts_2_diff import TS2DiffEncoder
from encoder.descending_separate import DescendingSeparateEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import EXP_RESULTS_DIR, DATA_DIR, DATA_NO_PERIOD_DIR, FIGURES_DIR
from typing import List
from encoder.descending_separate_v2 import DescendingSeparateEncoder


def plot_residual_distribution_with_bounds(
    residual_data, theoretical_bound, output_dir, dataset_name=""
):
    """
    Creates and saves a professional scatter plot of residual data with theoretical bounds.

    Args:
        residual_data (np.ndarray): The array of residual values.
        theoretical_bound (float): The calculated theoretical bound (e.g., your 'loss' value).
        output_dir (pathlib.Path): Directory to save the figure.
        dataset_name (str): Optional name of the dataset for the title/filename.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "svg.fonttype": "none",
        }
    )

    n = len(residual_data)
    sample_indices = np.arange(n)

    color_scatter = "#377eb8"
    outlier_color = "#ff7f00"
    color_bound = "#e41a1c"

    fig, ax = plt.subplots(figsize=(6, 4))

    sample_rate = 3
    ax.scatter(
        sample_indices[::sample_rate][
            np.abs(residual_data[::sample_rate]) < theoretical_bound
        ],
        residual_data[::sample_rate][
            np.abs(residual_data[::sample_rate]) < theoretical_bound
        ],
        s=10,
        alpha=0.5,
        color=color_scatter,
        edgecolor="none",
        label="Dense Residuals ($|R_k| < 2^D$)",
    )

    ax.scatter(
        sample_indices[::sample_rate][
            np.abs(residual_data[::sample_rate]) >= theoretical_bound
        ],
        residual_data[::sample_rate][
            np.abs(residual_data[::sample_rate]) >= theoretical_bound
        ],
        s=10,
        alpha=0.5,
        color=outlier_color,
        edgecolor="none",
        label="Long Tail ($|R_k| \\geq 2^D$)",
    )

    ax.axhline(
        y=theoretical_bound,
        color=color_bound,
        linestyle="--",
        linewidth=1.5,
        label=f"Example Bound ($\\pm 2^D, D={np.log2(theoretical_bound):.0f}$)",
    )
    ax.axhline(y=-theoretical_bound, color=color_bound, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Index ($k$)")
    ax.set_ylabel("Residual Value ($R_k$)")
    ax.legend()

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_abs_resid = np.max(np.abs(residual_data))
    ax.set_ylim(-max_abs_resid * 0.6, max_abs_resid * 0.6)
    ax.set_xlim(-0.01 * n, n * 1.01)

    fig.tight_layout()
    filename_suffix = (
        f"_{dataset_name.replace(' ', '_').lower()}" if dataset_name else ""
    )
    output_filename = output_dir / f"residual_distribution{filename_suffix}"
    plt.savefig(f"{output_filename}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{output_filename}.svg", format="svg", bbox_inches="tight")

    print(f"Residual distribution plot saved to {output_filename}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "dianwang_value_from2020-11-29to2020-12-06_10457.csv")
    data = df["value"].to_numpy(dtype="int64")
    n = data.shape[0]

    frequency_value = np.fft.fft(data)

    optimal_beta: int = 17

    quantized_real = np.round(np.real(frequency_value) / (2**optimal_beta)) * (
        2**optimal_beta
    )
    quantized_imag = np.round(np.imag(frequency_value) / (2**optimal_beta)) * (
        2**optimal_beta
    )
    frequency_value_quantized = quantized_real + 1j * quantized_imag

    reconstructed_data = np.round(
        np.real(np.fft.ifft(frequency_value_quantized))
    ).astype(np.int64)
    if len(reconstructed_data) != n:
        reconstructed_data = np.round(
            np.real(np.fft.ifft(frequency_value_quantized, n=n))
        ).astype(np.int64)

    residual = data - reconstructed_data

    e_freq_approx = (np.abs(frequency_value - frequency_value_quantized) ** 2).sum()
    expected_magnitude_approx = (1 / n) * np.sqrt(e_freq_approx)
    if expected_magnitude_approx < 1:
        expected_magnitude_approx = 1
    theoretical_bound = 2 ** (np.ceil(np.log2(expected_magnitude_approx)) + 1)
    print(
        f"Dummy residual max: {np.max(np.abs(residual))}, Theoretical Bound: {theoretical_bound}"
    )

    plot_residual_distribution_with_bounds(
        residual_data=residual,
        theoretical_bound=theoretical_bound,
        output_dir=FIGURES_DIR,
        dataset_name="Dianwang Sample",
    )
