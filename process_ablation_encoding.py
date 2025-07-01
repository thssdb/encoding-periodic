import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from config import EXP_RESULTS_DIR, DATA_DIR, FIGURES_DIR
from encoder.flea_ablation import FLEAEncoder
from exp_ablation_encoding import (
    RESULT_ABLATION_ENCODING,
    RESULT_NO_PERIOD_ABLATION_ENCODING,
)

if __name__ == "__main__":
    results = pd.read_csv(EXP_RESULTS_DIR / RESULT_ABLATION_ENCODING)
    results_no_period = pd.read_csv(
        EXP_RESULTS_DIR / RESULT_NO_PERIOD_ABLATION_ENCODING
    )
    results_all = pd.concat([results, results_no_period], ignore_index=True)

    results_all["compression_ratio"] = (
        results_all["data_size"] / results_all["stream_size"]
    )

    result = (
        results_all.groupby(["simple_frequency", "simple_residual", "dataset"])
        .agg(compression_ratio=("compression_ratio", "mean"))
        .groupby(["simple_frequency", "simple_residual"])
        .agg(compression_ratio=("compression_ratio", "mean"))
    )

    best_compression_ratio = result["compression_ratio"].max()
    result["performance_drop"] = (
        (best_compression_ratio - result["compression_ratio"]) / best_compression_ratio
    ) * 100

    result["compression_ratio"] = result["compression_ratio"].round(2)
    result["performance_drop"] = result["performance_drop"].round(2)

    print(result)
