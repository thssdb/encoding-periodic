import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATA_DIR, FIGURES_DIR
from encoder.flea_ablation import FLEAEncoder
from algorithm.separate_storage import get_optimal_d, get_bit_length_cnt


def quantize_spectrum(spectrum, beta):
    """Simple quantization based on beta."""
    delta = 2**beta
    return np.round(spectrum / delta) * delta


def get_reconstructed_signal(original_signal, beta):
    """Performs FFT, quantization, and IFFT to get the reconstructed signal."""
    n = len(original_signal)
    freq_spectrum = np.fft.fft(original_signal)
    quantized_dequantized_spectrum = quantize_spectrum(freq_spectrum, beta)
    reconstructed = np.fft.ifft(quantized_dequantized_spectrum)
    return np.real(reconstructed)


def generate_plots_for_ppt(
    time_series_data, frequency_count, beta_param, p_param, d_param, output_dir
):
    """
    Generates four separate, clean plots with annotations for assembly in PPT.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 18,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "axes.titlesize": 11,
            "svg.fonttype": "none",
        }
    )

    time_domain_color = "#377eb8"
    frequency_domain_color = "#9467bd"
    residual_color = "#2ca02c"
    parameter_color = "#e41a1c"

    n = len(time_series_data)
    freq_spectrum = np.fft.fft(time_series_data)
    quantized_spectrum = quantize_spectrum(freq_spectrum, beta_param)
    freq_axis = np.arange(n // 2)
    original_magnitudes = np.abs(np.imag(freq_spectrum))[: n // 2]
    quantized_magnitudes = np.abs(np.imag(quantized_spectrum))[: n // 2]
    reconstructed_signal = get_reconstructed_signal(time_series_data, beta_param)
    residual_signal = time_series_data - np.round(reconstructed_signal).astype(np.int64)

    width = 2.5

    fig_a, ax_a = plt.subplots(figsize=(width, width * (3 / 4)))
    ax_a.plot(np.arange(n), time_series_data, color=time_domain_color, linewidth=1.2)
    ax_a.set_xlim(0, n)
    ax_a.set_title("")
    ax_a.set_xlabel("Time Step")
    ax_a.set_ylabel("Value")
    ax_a.set_xticklabels([])
    ax_a.set_yticklabels([])
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    fig_a.tight_layout(pad=0.1)
    fig_a.savefig(
        output_dir / "plot_a_timeseries.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig_a.savefig(
        output_dir / "plot_a_timeseries.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(width, width * (3 / 4)))
    ax_b.plot(
        freq_axis[:frequency_count],
        original_magnitudes[:frequency_count],
        color=frequency_domain_color,
        linewidth=1.2,
    )
    quantization_threshold = 2**beta_param
    ax_b.axhline(
        y=quantization_threshold,
        color=parameter_color,
        linestyle="--",
        linewidth=1.5,
    )
    ax_b.text(
        frequency_count * 0.8,
        quantization_threshold * 3,
        "$2^\\beta$",
        color=parameter_color,
        ha="left",
        va="center",
        fontsize=24,
    )
    ax_b.set_xlim(0, frequency_count)
    ax_b.set_ylim(bottom=0)
    ax_b.set_title("")
    ax_b.set_xlabel("Frequency Index")
    ax_b.set_ylabel("Magnitude")
    ax_b.set_xticklabels([])
    ax_b.set_yticklabels([])
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    fig_b.tight_layout(pad=0.1)
    fig_b.savefig(
        output_dir / "plot_b_spectrum.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig_b.savefig(
        output_dir / "plot_b_spectrum.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(width, width * (3 / 4)))
    markerline, stemlines, baseline = ax_c.stem(
        freq_axis[:frequency_count],
        quantized_magnitudes[:frequency_count],
        linefmt="grey",
        markerfmt="o",
        basefmt="k-",
    )
    plt.setp(markerline, "markersize", 3, "color", frequency_domain_color)
    plt.setp(stemlines, "linewidth", 0.8)
    plt.setp(baseline, "linewidth", 1)
    ax_c.axvline(x=p_param, color=parameter_color, linestyle="--", linewidth=1.5)
    ax_c.text(
        p_param * 1.1,
        ax_c.get_ylim()[1] * 0.8,
        "$p$",
        color=parameter_color,
        ha="left",
        va="center",
        fontsize=24,
    )
    ax_c.set_xlim(0, frequency_count)
    ax_c.set_ylim(bottom=0)
    ax_c.set_title("")
    ax_c.set_xlabel("Frequency Index")
    ax_c.set_ylabel("Magnitude")
    ax_c.set_xticklabels([])
    ax_c.set_yticklabels([])
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    fig_c.tight_layout(pad=0.1)
    fig_c.savefig(
        output_dir / "plot_c_quantized.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig_c.savefig(
        output_dir / "plot_c_quantized.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig_c)

    fig_d, ax_d = plt.subplots(figsize=(width, width * (3 / 4)))
    indices = np.arange(n)
    residual_threshold = 2**d_param
    ax_d.scatter(
        indices[np.abs(residual_signal) < residual_threshold],
        residual_signal[np.abs(residual_signal) < residual_threshold],
        s=10,
        alpha=0.5,
        color=residual_color,
        edgecolor="none",
        label="Dense Residuals ($|R_k| < 2^D$)",
    )
    ax_d.scatter(
        indices[np.abs(residual_signal) >= residual_threshold],
        residual_signal[np.abs(residual_signal) >= residual_threshold],
        s=10,
        alpha=0.5,
        color=residual_color,
        edgecolor="none",
        label="Outliers ($|R_k| \\geq 2^D$)",
    )
    ax_d.axhline(
        y=residual_threshold, color=parameter_color, linestyle="--", linewidth=1.5
    )
    ax_d.axhline(
        y=-residual_threshold, color=parameter_color, linestyle="--", linewidth=1.5
    )
    ax_d.text(
        n * 0.8,
        residual_threshold,
        "$2^D$",
        color=parameter_color,
        ha="left",
        va="bottom",
        fontsize=24,
    )
    ax_d.set_xlim(0, n)
    ax_d.set_ylim(bottom=-residual_threshold * 2, top=residual_threshold * 2)
    ax_d.set_title("")
    ax_d.set_xlabel("Time Step")
    ax_d.set_ylabel("Residual Value")
    ax_d.set_xticklabels([])
    ax_d.set_yticklabels([])
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    fig_d.tight_layout(pad=0.1)
    fig_d.savefig(
        output_dir / "plot_d_residual.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig_d.savefig(
        output_dir / "plot_d_residual.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig_d)

    print(f"Generated 4 clean plots in '{output_dir}' for PPT assembly.")


if __name__ == "__main__":
    n = 900
    frequency_count = 100
    time = np.arange(n)
    df = pd.read_csv(DATA_DIR / "liantong_data_from2018-12-19to2019-01-31_8205.csv")
    data = df["value"].to_numpy(dtype=np.int64)[:n]
    frequency = np.fft.fft(data)
    frequency_real = np.real(frequency).astype(np.int64)
    frequency_imag = np.imag(frequency).astype(np.int64)

    encoder = FLEAEncoder()

    BETA_PARAM = encoder.get_optimal_beta(n, frequency_real, frequency_imag)
    frequency_real_quantized = np.round(frequency_real / (2**BETA_PARAM)).astype(
        np.int64
    )
    frequency_imag_quantized = np.round(frequency_imag / (2**BETA_PARAM)).astype(
        np.int64
    )

    frequency_quantized = frequency_real_quantized.astype(np.complex128) * (
        2**BETA_PARAM
    ) + 1j * frequency_imag_quantized.astype(np.complex128) * (2**BETA_PARAM)
    residual = data - np.round(np.real(np.fft.ifft(frequency_quantized))).astype(
        np.int64
    )
    residual_unsigned = np.abs(residual).astype(np.uint64)

    P_PARAM = encoder.laminar_hybrid_encoder.partition(frequency_imag_quantized)
    D_PARAM = get_optimal_d(n, get_bit_length_cnt(residual_unsigned))

    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    generate_plots_for_ppt(
        data, frequency_count, BETA_PARAM, P_PARAM, D_PARAM, FIGURES_DIR
    )
