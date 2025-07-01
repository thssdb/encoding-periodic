import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from config import EXP_RESULTS_DIR, DATA_DIR, DATA_NO_PERIOD_DIR, FIGURES_DIR
from encoder.flea_ablation import FLEAEncoder
from exp_ablation_beta import RESULT_ABLATION, RESULT_NO_PERIOD_ABLATION
from exp_ablation_p import RESULT_ABLATION_P, RESULT_NO_PERIOD_ABLATION_P


def plot_beta_analysis_figure(
    df_avg_results, df_single_instance_results, optimal_avg_cr, output_dir
):
    pass


def plot_p_analysis_figure(
    df_avg_p_results, optimal_avg_cr, df_single_instance_p, output_dir
):
    """
    Generates the combined figure for split point p analysis.
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
    p_ratio_values = df_avg_p_results.index
    avg_cr_values = df_avg_p_results["compression_ratio"]

    color_fixed_p = "#377eb8"
    color_optimal = "#e41a1c"

    ax1.plot(
        p_ratio_values,
        avg_cr_values,
        "s-",
        color=color_fixed_p,
        label=r"Preset $p/n$",
    )
    ax1.axhline(
        optimal_avg_cr,
        color=color_optimal,
        linestyle="--",
        label=r"Optimal $p^*$",
    )

    ax1.set_title("(b) Performance on All Datasets")
    ax1.set_xticks(p_ratio_values)
    ax1.set_xlabel(r"Split Index Ratio ($p/n$)")
    ax1.set_ylabel("Average Compression Ratio")
    ax1.legend(loc="lower center")
    ax1.set_ylim(bottom=5, top=6.5)

    ax0 = axes[0]
    n = df_single_instance_p.shape[0] - 1
    p_indices = (p_ratio_values * n).astype(int)

    color_dense = "#8da0cb"
    color_sparse = "#fc8d62"
    color_total = "#66c2a5"

    ax0.plot(
        p_indices / n,
        df_single_instance_p["prefix_encoding_length"][p_indices],
        "^--",
        color=color_dense,
        label="$L(\\hat{\\mathbf{F}}[0:p])$",
    )
    ax0.plot(
        p_indices / n,
        df_single_instance_p["suffix_encoding_length"][p_indices],
        "v:",
        color=color_sparse,
        label="$L(\\hat{\\mathbf{F}}[p:n])$",
    )
    ax0.plot(
        p_indices / n,
        df_single_instance_p["total_encoding_length"][p_indices],
        "s-",
        color=color_total,
        linewidth=2,
        label="Total Cost",
    )

    best_idx = df_single_instance_p["total_encoding_length"][p_indices].idxmin()
    best_cost = df_single_instance_p["total_encoding_length"][p_indices].min()
    ax0.plot(
        best_idx / n,
        best_cost,
        "*",
        color=color_optimal,
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=10,
    )

    max_y = df_single_instance_p["total_encoding_length"].max()

    ax0.set_title("(a) Determination on Sample Series", x=0.47)
    ax0.set_xticks(p_ratio_values)
    ax0.set_xlabel(r"Split Index Ratio ($p/n$)")
    ax0.set_ylabel("Encoding Cost (bits)")
    ax0.legend(loc="right", bbox_to_anchor=(1, 0.33), fontsize=11)
    ax0.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax0.yaxis.get_offset_text()
    offset_text.set_position((-0.08, 1.02))
    offset_text.set_horizontalalignment("right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":")

    fig.tight_layout(pad=0.5)

    output_filename = output_dir / "p_analysis_combined"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.eps", format="eps")
    print(f"Split point analysis plot saved to {output_filename}.pdf/.eps")


def get_p_split_example_data():
    df = pd.read_csv(DATA_DIR / "guoshou_value_from2020-01-02to2020-01-19_7468.csv")
    data = df["value"].to_numpy(dtype="int64")

    encoder = FLEAEncoder(calculate_partition=True)
    result = encoder.exp(data)

    return pd.DataFrame(
        {
            "prefix_encoding_length": encoder.partition_prefix_encoding_length,
            "suffix_encoding_length": encoder.partition_suffix_encoding_length,
            "total_encoding_length": encoder.partition_prefix_encoding_length
            + encoder.partition_suffix_encoding_length,
        }
    )


if __name__ == "__main__":
    results_p = pd.read_csv(EXP_RESULTS_DIR / RESULT_ABLATION_P)
    results_no_period_p = pd.read_csv(EXP_RESULTS_DIR / RESULT_NO_PERIOD_ABLATION_P)
    results_all_p = pd.concat([results_p, results_no_period_p], ignore_index=True)

    results_all_p["compression_ratio"] = (
        results_all_p["data_size"] / results_all_p["stream_size"]
    )

    fixed_p_runs = results_all_p[results_all_p["is_given_p"] == True]
    avg_fixed_p_cr = (
        fixed_p_runs.groupby(["given_p", "dataset"])
        .agg(compression_ratio=("compression_ratio", "mean"))
        .groupby("given_p")
        .agg(compression_ratio=("compression_ratio", "mean"))
    )

    optimal_flea_runs = results_all_p[results_all_p["is_given_p"] == False]
    overall_optimal_flea_cr = (
        optimal_flea_runs.groupby("dataset")
        .agg(compression_ratio=("compression_ratio", "mean"))
        .agg(compression_ratio=("compression_ratio", "mean"))
    )["compression_ratio"].iloc[0]

    single_instance_p_results = get_p_split_example_data()

    plot_p_analysis_figure(
        df_avg_p_results=avg_fixed_p_cr,
        optimal_avg_cr=overall_optimal_flea_cr,
        df_single_instance_p=single_instance_p_results,
        output_dir=FIGURES_DIR,
    )
