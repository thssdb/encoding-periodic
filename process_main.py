import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import EXP_RESULTS_DIR
from exp_main import RESULT_MAIN, RESULT_NO_PERIOD_MAIN

ALGORITHM_NAME_MAP = {"Descending-Separate-V2": "FLEA", "Buff": "BUFF", "Hire": "HIRE"}
ALGORITHM_ORDER = ["FLEA", "Sprintz", "Chimp", "BUFF", "Gorilla", "HIRE", "RLE"]
DATASET_NAME_MAP = {
    "dianwang": "Grid",
    "guoshou": "Safe",
    "liantong": "Mobile",
    "yinlian": "Bank",
    "temperature": "Temperature",
    "volt": "Volt",
    "gps": "GPS",
    "power": "Power",
    "stock": "Stock",
}
PERIODIC_DATASETS = ["Temperature", "Volt", "Grid", "Safe", "Mobile", "Bank"]
APERIODIC_DATASETS = ["Power", "GPS", "Stock", "Chemistry"]
DATASET_ORDER = PERIODIC_DATASETS + APERIODIC_DATASETS


def highlight_max_and_second(s, float_format="{:.2f}"):
    if not pd.api.types.is_numeric_dtype(s):
        return s
    unique_sorted = s.dropna().unique()
    unique_sorted.sort()
    max_val = unique_sorted[-1] if len(unique_sorted) > 0 else -np.inf
    second_max_val = unique_sorted[-2] if len(unique_sorted) > 1 else -np.inf
    output = []
    for val in s:
        if pd.isna(val):
            output.append("")
        elif val == max_val:
            output.append(f"\\textbf{{{float_format.format(val)}}}")
        elif val == second_max_val:
            output.append(f"\\underline{{{float_format.format(val)}}}")
        else:
            output.append(float_format.format(val))
    return output


def process_results_for_publication(results: pd.DataFrame):
    results_processed = results.copy()
    results_processed["encoder"] = results_processed["encoder"].replace(
        ALGORITHM_NAME_MAP
    )
    results_processed["encoder"] = pd.Categorical(
        results_processed["encoder"], categories=ALGORITHM_ORDER, ordered=True
    )
    results_processed["dataset"] = results_processed["dataset"].replace(
        DATASET_NAME_MAP
    )
    results_processed["dataset"] = pd.Categorical(
        results_processed["dataset"], categories=DATASET_ORDER, ordered=True
    )
    results_processed["compression_ratio"] = (
        results_processed["data_size"] / results_processed["stream_size"]
    )
    bytes_to_mib = 1024 * 1024
    results_processed["encoding_throughput"] = (
        results_processed["data_size"] / bytes_to_mib
    ) / results_processed["encoding_time"]
    results_processed["decoding_throughput"] = (
        results_processed["data_size"] / bytes_to_mib
    ) / results_processed["decoding_time"]
    results_grouped = results_processed.groupby(
        ["encoder", "dataset"], observed=False
    ).agg(
        {
            "compression_ratio": "mean",
            "encoding_throughput": "mean",
            "decoding_throughput": "mean",
        }
    )

    def create_metric_table(metric_name):
        pivot = results_grouped.pivot_table(
            index="encoder", columns="dataset", values=metric_name, observed=False
        ).reindex(ALGORITHM_ORDER)[DATASET_ORDER]
        pivot["Periodic Avg."] = pivot[PERIODIC_DATASETS].mean(axis=1)
        pivot["Aperiodic Avg."] = pivot[APERIODIC_DATASETS].mean(axis=1)
        pivot["Overall Avg."] = pivot[DATASET_ORDER].mean(axis=1)
        pivot = pivot[
            PERIODIC_DATASETS
            + ["Periodic Avg."]
            + APERIODIC_DATASETS
            + ["Aperiodic Avg."]
            + ["Overall Avg."]
        ]
        return pivot

    cr_table_raw = create_metric_table("compression_ratio")
    et_table_raw = create_metric_table("encoding_throughput")
    dt_table_raw = create_metric_table("decoding_throughput")

    formatted_cr_table = cr_table_raw.apply(highlight_max_and_second, axis=0)

    formatted_et_table = et_table_raw.map(lambda x: f"{x:.2f}")
    formatted_dt_table = dt_table_raw.map(lambda x: f"{x:.2f}")

    return {
        "compression_ratio_formatted": formatted_cr_table,
        "compression_ratio_raw": cr_table_raw,
        "encoding_throughput_formatted": formatted_et_table,
        "encoding_throughput_raw": et_table_raw,
        "decoding_throughput_formatted": formatted_dt_table,
        "decoding_throughput_raw": dt_table_raw,
    }


def plot_avg_compression_ratio(df_raw, output_dir):
    """
    Creates and saves a professional grouped bar chart with centered, non-rotated labels.

    Args:
        df_raw (pd.DataFrame): The raw (unformatted) pivot table containing average columns.
        output_dir (str or Path): The path to save the figure file.
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
            "hatch.linewidth": 0.6,
        }
    )

    df_plot = df_raw[["Periodic Avg.", "Aperiodic Avg."]].copy()
    algorithms = df_plot.index.tolist()
    x = np.arange(len(algorithms))
    width = 0.35

    color_periodic = "#8da0cb"
    color_aperiodic = "#fc8d62"
    hatch_periodic = "///"
    hatch_aperiodic = "..."

    fig, ax = plt.subplots(figsize=(6.5, 6.5 * (9 / 16)))

    rects1 = ax.bar(
        x - width / 2,
        df_plot["Periodic Avg."],
        width,
        label="Periodic Datasets",
        color=color_periodic,
        edgecolor="black",
        hatch=hatch_periodic,
    )

    rects2 = ax.bar(
        x + width / 2,
        df_plot["Aperiodic Avg."],
        width,
        label="Aperiodic Datasets",
        color=color_aperiodic,
        edgecolor="black",
        hatch=hatch_aperiodic,
    )

    ax.set_ylabel("Average Compression Ratio")

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=0, ha="center")

    ax.legend()

    ax.bar_label(rects1, padding=3, fmt="%.2f", fontsize=8)
    ax.bar_label(rects2, padding=3, fmt="%.2f", fontsize=8)

    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, df_plot.to_numpy().max() * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    output_filename = output_dir / "avg_compression_ratio"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.eps", format="eps")

    print(f"Final compression ratio plot saved to {output_filename}.pdf/.eps")


def plot_avg_throughput(df_encoding_raw, df_decoding_raw, output_dir):
    """
    Creates a grouped bar chart comparing average Encoding vs. Decoding throughput.

    Args:
        df_encoding_raw (pd.DataFrame): Raw pivot table for encoding throughput.
        df_decoding_raw (pd.DataFrame): Raw pivot table for decoding throughput.
        output_dir (str or Path): The path to save the figure file.
    """
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
            "hatch.linewidth": 0.6,
        }
    )

    avg_encoding = df_encoding_raw["Overall Avg."]
    avg_decoding = df_decoding_raw["Overall Avg."]

    algorithms = avg_encoding.index.tolist()
    x = np.arange(len(algorithms))
    width = 0.35

    color_encoding = "#66c2a5"
    color_decoding = "#fdb462"
    hatch_encoding = "xx"
    hatch_decoding = "++"

    fig, ax = plt.subplots(figsize=(6.5, 6.5 * (9 / 16)))

    rects1 = ax.bar(
        x - width / 2,
        avg_encoding,
        width,
        label="Encoding",
        color=color_encoding,
        edgecolor="black",
        hatch=hatch_encoding,
    )

    rects2 = ax.bar(
        x + width / 2,
        avg_decoding,
        width,
        label="Decoding",
        color=color_decoding,
        edgecolor="black",
        hatch=hatch_decoding,
    )

    ax.set_ylabel("Average Throughput (MiB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=0, ha="center")
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt="%.2f", fontsize=8)
    ax.bar_label(rects2, padding=3, fmt="%.2f", fontsize=8)

    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    max_val = max(avg_encoding.max(), avg_decoding.max())
    ax.set_ylim(0, max_val * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    output_filename = output_dir / "avg_throughput_comparison"
    plt.savefig(f"{output_filename}.pdf", format="pdf")
    plt.savefig(f"{output_filename}.eps", format="eps")

    print(f"Average throughput comparison plot saved to {output_filename}.pdf/.eps")


if __name__ == "__main__":
    results = pd.read_csv(EXP_RESULTS_DIR / RESULT_MAIN)
    results["periodic"] = True
    results_no_period = pd.read_csv(EXP_RESULTS_DIR / RESULT_NO_PERIOD_MAIN)
    results_no_period["periodic"] = False
    results_all = pd.concat([results, results_no_period], ignore_index=True)

    processed_data = process_results_for_publication(results_all)

    processed_data["compression_ratio_formatted"].to_latex(
        EXP_RESULTS_DIR / "compression_ratio_table_content.tex",
        escape=False,
        header=True,
        index=True,
    )
    print("CR table content saved.")

    processed_data["encoding_throughput_formatted"].to_latex(
        EXP_RESULTS_DIR / "encoding_throughput_table_content.tex",
        escape=False,
        header=True,
        index=True,
    )
    print("Encoding throughput table content saved.")

    processed_data["decoding_throughput_formatted"].to_latex(
        EXP_RESULTS_DIR / "decoding_throughput_table_content.tex",
        escape=False,
        header=True,
        index=True,
    )
    print("Decoding throughput table content saved.")

    plot_avg_compression_ratio(
        df_raw=processed_data["compression_ratio_raw"], output_dir=EXP_RESULTS_DIR
    )

    plot_avg_throughput(
        df_encoding_raw=processed_data["encoding_throughput_raw"],
        df_decoding_raw=processed_data["decoding_throughput_raw"],
        output_dir=EXP_RESULTS_DIR,
    )
