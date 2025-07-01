import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from config import EXP_RESULTS_DIR, DATA_DIR, DATA_NO_PERIOD_DIR, FIGURES_DIR
from encoder.flea_ablation import (
    FLEAEncoder,
)
from exp_ablation_beta import RESULT_ABLATION, RESULT_NO_PERIOD_ABLATION


def plot_beta_analysis_figure(
    df_avg_results, df_single_instance_results, optimal_avg_cr, output_dir
):
    """
    Generates the combined figure for beta analysis, showing both aggregate performance
    and the underlying mechanism on a single instance.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "#d0d0d0",
            "lines.linewidth": 1.5,
            "lines.markersize": 7.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))

    ax1 = axes[1]
    beta_values_agg = df_avg_results.index
    avg_cr_values = df_avg_results["compression_ratio"]

    color_fixed_beta = "#377eb8"
    color_optimal = "#e41a1c"

    ax1.plot(
        beta_values_agg,
        avg_cr_values,
        "s-",
        color=color_fixed_beta,
        label=r"Preset $\beta$",
    )
    ax1.axhline(
        optimal_avg_cr,
        color=color_optimal,
        linestyle="--",
        label=r"Optimal $\beta^*$",
    )

    ax1.set_title(r"(b) Performance on All Datasets")
    ax1.set_xticks(beta_values_agg)
    ax1.set_xlabel(r"Quantization Level ($\beta$)")
    ax1.set_ylabel("Average Compression Ratio")
    ax1.legend(loc="lower center")
    ax1.set_ylim(bottom=5, top=6.5)

    ax0 = axes[0]
    beta_values_single = df_single_instance_results["beta"]

    color_freq = "#8da0cb"
    color_resid = "#fc8d62"
    color_total = "#66c2a5"

    ax0.plot(
        beta_values_single,
        df_single_instance_results["frequency_encoding_length"],
        "^--",
        color=color_freq,
        label=r"$L(\hat{\mathbf{F}}_{\beta})$",
    )
    ax0.plot(
        beta_values_single,
        df_single_instance_results["residual_encoding_length"],
        "v:",
        color=color_resid,
        label=r"$L(\mathbf{R}_{\beta})$",
    )
    ax0.plot(
        beta_values_single,
        df_single_instance_results["total_encoding_length"],
        "s-",
        color=color_total,
        linewidth=2,
        label="Total Cost",
    )

    best_beta = df_single_instance_results["beta"].iloc[
        df_single_instance_results["total_encoding_length"].idxmin()
    ]
    best_cost = df_single_instance_results["total_encoding_length"].min()
    ax0.plot(
        best_beta,
        best_cost,
        "*",
        color=color_optimal,
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=10,
    )

    ax0.set_title("(a) Determination on Sample Series", x=0.45)
    ax0.set_xticks(beta_values_agg)
    ax0.set_xlabel(r"Quantization Level ($\beta$)")
    ax0.set_ylabel("Encoding Cost (bits)")
    ax0.legend(loc="right", bbox_to_anchor=(1, 0.4))
    ax0.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax0.yaxis.get_offset_text()
    offset_text.set_position((-0.12, 1.02))
    offset_text.set_horizontalalignment("right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":")

    fig.tight_layout(pad=0.5)

    output_filename = output_dir / "beta_analysis_combined"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.eps", format="eps")
    print(f"Beta analysis plot saved to {output_filename}.pdf/.eps")


def get_example_data():
    df = pd.read_csv(DATA_DIR / "liantong_data_from2018-12-19to2019-01-31_8205.csv")
    data = df["value"].to_numpy(dtype="int64")

    beta_values = np.arange(10, 21, 2)
    results: List[pd.DataFrame] = []
    for beta in beta_values:
        encoder = FLEAEncoder(is_given_beta=True, given_beta=int(beta))
        result = encoder.exp(data)
        results.append(result)
    result = pd.concat(results, ignore_index=True)

    frequency_encoding_length = result["frequency_encoding_length"].to_numpy(
        dtype="int64"
    )
    residual_encoding_length = result["residual_encoding_length"].to_numpy(
        dtype="int64"
    )
    total_encoding_length = frequency_encoding_length + residual_encoding_length

    return pd.DataFrame(
        {
            "beta": beta_values,
            "frequency_encoding_length": frequency_encoding_length,
            "residual_encoding_length": residual_encoding_length,
            "total_encoding_length": total_encoding_length,
        }
    )


if __name__ == "__main__":
    results = pd.read_csv(EXP_RESULTS_DIR / RESULT_ABLATION)
    results_no_period = pd.read_csv(EXP_RESULTS_DIR / RESULT_NO_PERIOD_ABLATION)
    results_all = pd.concat([results, results_no_period], ignore_index=True)

    results_all["compression_ratio"] = (
        results_all["data_size"] / results_all["stream_size"]
    )

    fixed_beta_runs = results_all[results_all["is_given_beta"] == True].copy()

    avg_fixed_beta_cr = (
        fixed_beta_runs.groupby(["given_beta", "dataset"])
        .agg(compression_ratio=("compression_ratio", "mean"))
        .groupby("given_beta")
        .agg(compression_ratio=("compression_ratio", "mean"))
    )

    optimal_flea_runs = results_all[results_all["is_given_beta"] == False]

    overall_optimal_flea_cr = (
        optimal_flea_runs.groupby("dataset")
        .agg(compression_ratio=("compression_ratio", "mean"))
        .agg(compression_ratio=("compression_ratio", "mean"))
    )["compression_ratio"].iloc[0]

    single_instance_results = get_example_data()

    plot_beta_analysis_figure(
        df_avg_results=avg_fixed_beta_cr,
        df_single_instance_results=single_instance_results,
        optimal_avg_cr=overall_optimal_flea_cr,
        output_dir=FIGURES_DIR,
    )
