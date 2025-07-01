import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from config import EXP_RESULTS_DIR, DATA_DIR, DATA_NO_PERIOD_DIR, FIGURES_DIR
from encoder.flea_ablation import FLEAEncoder
from exp_ablation_d import RESULT_ABLATION_D, RESULT_NO_PERIOD_ABLATION_D


def plot_d_analysis_figure(
    df_avg_d_results, optimal_avg_cr, df_single_instance_d, output_dir
):
    """
    Generates the combined figure for residual partition D analysis.
    The right subplot shows the cost trade-off for D on a single instance.
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
    d_values_agg = df_avg_d_results.index
    avg_cr_values = df_avg_d_results["compression_ratio"]
    color_fixed_d = "#377eb8"
    color_optimal = "#e41a1c"
    ax1.plot(
        d_values_agg,
        avg_cr_values,
        "s-",
        color=color_fixed_d,
        label=r"Preset $D$",
    )
    ax1.axhline(
        optimal_avg_cr,
        color=color_optimal,
        linestyle="--",
        label=r"Optimal $D^*$",
    )
    ax1.set_title("(b) Performance on All Datasets")
    ax1.set_xticks(d_values_agg)
    ax1.set_xlabel(r"Residual Cut Bit ($D$)")
    ax1.set_ylabel("Average Compression Ratio")
    ax1.legend(loc="lower center")
    ax1.set_ylim(bottom=2.5, top=6.5)

    ax0 = axes[0]
    d_values_single = df_single_instance_d["d"]
    low_bit_cost = df_single_instance_d["low_bit_encoding_length"]
    high_bit_cost = df_single_instance_d["residual_encoding_length"] - low_bit_cost
    total_cost = df_single_instance_d["residual_encoding_length"]

    color_low_bit = "#8da0cb"
    color_high_bit = "#fc8d62"
    color_total = "#66c2a5"
    color_opt_marker = "#e41a1c"

    ax0.plot(
        d_values_single, low_bit_cost, "^--", color=color_low_bit, label="Low-Bit Cost"
    )
    ax0.plot(
        d_values_single,
        high_bit_cost,
        "v:",
        color=color_high_bit,
        label="High-Bit Cost",
    )
    ax0.plot(
        d_values_single,
        total_cost,
        "s-",
        color=color_total,
        linewidth=2,
        label="Total Cost",
    )

    best_idx = total_cost.idxmin()
    optimal_d = d_values_single[best_idx]
    min_total_cost = total_cost[best_idx]
    ax0.plot(
        optimal_d,
        min_total_cost,
        "*",
        color=color_opt_marker,
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=10,
    )

    ax0.set_title("(a) Determination on Sample Series", x=0.47)
    ax0.set_xticks(d_values_agg)
    ax0.set_xlabel(r"Residual Cut Bit ($D$)")
    ax0.set_ylabel("Encoding Cost (bits)")
    ax0.legend(loc="right", bbox_to_anchor=(1, 0.3))
    ax0.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax0.yaxis.get_offset_text()
    offset_text.set_position((-0.08, 1.02))
    offset_text.set_horizontalalignment("right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":")

    fig.tight_layout(pad=0.5)

    output_filename = output_dir / "d_analysis_combined"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.eps", format="eps")
    print(f"Residual parameter D analysis plot saved to {output_filename}.pdf/.eps")


def get_d_split_example_data():
    """Generates the data needed for the right subplot of D analysis."""
    df = pd.read_csv(DATA_NO_PERIOD_DIR / "gps_32.csv")
    data = df["value"].to_numpy(dtype="int64")

    d_range = np.arange(6, 18, 2)
    results: List[pd.DataFrame] = []

    for d in d_range:
        encoder = FLEAEncoder(is_given_d=True, given_d=int(d))
        result = encoder.exp(data)
        results.append(result)

    results_df = pd.concat(results, ignore_index=True)
    results_df["d"] = d_range
    results_df_print = results_df[
        [
            "d",
            "low_bit_encoding_length",
            "residual_encoding_length",
        ]
    ]
    results_df_print["high_bit_encoding_length"] = (
        results_df_print["residual_encoding_length"]
        - results_df_print["low_bit_encoding_length"]
    )
    print(
        results_df_print[
            [
                "d",
                "low_bit_encoding_length",
                "high_bit_encoding_length",
                "residual_encoding_length",
            ]
        ]
    )

    return results_df


if __name__ == "__main__":
    results_d = pd.read_csv(EXP_RESULTS_DIR / RESULT_ABLATION_D)
    results_no_period_d = pd.read_csv(EXP_RESULTS_DIR / RESULT_NO_PERIOD_ABLATION_D)
    results_all_d = pd.concat([results_d, results_no_period_d], ignore_index=True)
    results_all_d["compression_ratio"] = (
        results_all_d["data_size"] / results_all_d["stream_size"]
    )
    fixed_d_runs = results_all_d[results_all_d["is_given_d"] == True]
    avg_fixed_d_cr = (
        fixed_d_runs.groupby(["given_d", "dataset"])
        .agg(compression_ratio=("compression_ratio", "mean"))
        .groupby("given_d")
        .agg(compression_ratio=("compression_ratio", "mean"))
    )
    optimal_flea_runs = results_all_d[results_all_d["is_given_d"] == False]
    overall_optimal_flea_cr = (
        optimal_flea_runs.groupby("dataset")
        .agg(compression_ratio=("compression_ratio", "mean"))
        .agg(compression_ratio=("compression_ratio", "mean"))
    )["compression_ratio"].iloc[0]

    single_instance_d_results = get_d_split_example_data()

    plot_d_analysis_figure(
        df_avg_d_results=avg_fixed_d_cr,
        optimal_avg_cr=overall_optimal_flea_cr,
        df_single_instance_d=single_instance_d_results,
        output_dir=FIGURES_DIR,
    )
